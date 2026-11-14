import os
import json
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import time
import torch
import random
import argparse
import numpy as np
from torchvision import models,transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss,resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching

# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
from model.HMmamba import HMmamba

from calflops import calculate_flops

# 启用 PyTorch 的自动梯度异常检测功能
# 当设置为 True 时，会在前向传播和反向传播期间检测可能的数值错误（如 NaN、inf 等）
# 检测到异常时会打印详细的错误信息，帮助调试
# 这会增加计算开销，通常只在调试时使用
torch.autograd.set_detect_anomaly(True)

# 生成当前时间的格式化字符串 用于创建带时间戳的日志文件或实验记录
# time.localtime() 获取当前本地时间，返回一个时间元组
# time.strftime() 将时间格式化为指定字符串：
# %y：两位数的年份  %m：两位数的月份 %d：两位数的日期 %H：24小时制的小时 %M：分钟
time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())




def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting 随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int,default=3) # 数据集索引
    parser.add_argument('--data_set_path',type=str,default='./data') # 数据集存储路径
    parser.add_argument('--work_dir',type=str,default='./') # 工作目录
    parser.add_argument('--lr', type=float, default=0.0003) # 学习率
    parser.add_argument('--max_epoch', type=int, default=200) # 最大训练轮数
    parser.add_argument('--train_samples', type=int, default=30) # 每个类别的训练样本数
    parser.add_argument('--val_samples', type=int, default=10) # 每个类别的验证样本数
    parser.add_argument('--exp_name', type=str, default='RUNS') # 实验名称用于保存结果
    parser.add_argument('--record_computecost',type=bool,default=True) # 是否记录计算成本

    args = parser.parse_args()
    return args

# 参数解析：
# 设备选择：检查是否有可用的GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser() # 解析命令行参数
record_computecost = args.record_computecost #是否记录计算成本
exp_name = args.exp_name # 实验名称
seed_list = [0,1,2,3,4,5,6,7,8,9]  # 随机种子列表，用于多次实验取平均
# seed_list = [0,1,2]  # 随机种子列表，用于多次实验取平均
# seed_list = [0]  #

num_list = [args.train_samples, args.val_samples] # 训练和验证样本数

dataset_index = args.dataset_index # 数据集索引

max_epoch = args.max_epoch # 最大训练轮数
learning_rate = args.lr # 学习率

net_name = 'ACmamba' # 模型名称


# 参数字典：将关键参数存储到字典中，便于后续使用或记录
paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list}


# 数据集配置：
                      # 0     1     2       3        4         5    6
data_set_name_list = ['PU', 'HC', 'HH', 'Houston','Botswana','IP','LK','SA']
data_set_name = data_set_name_list[dataset_index] # 根据索引选择数据集

if data_set_name in ['n','Botswana']:
    split_image = True   #分块
else:
    split_image = False  #不分块


# 数据预处理：
transform = transforms.Compose([
    # transforms.Resize((2048, 1024)),
    transforms.ToTensor(), # 将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor（维度为 [C, H, W]）

    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
])


if __name__ == '__main__':
    data_set_path = args.data_set_path # 数据集路径
    work_dir = args.work_dir # 工作目录
    setting_name = 'tr{}val{}'.format(str(args.train_samples),str(args.val_samples)) + '_lr{}'.format(str(learning_rate))
    dataset_name = data_set_name # 数据集名称
    exp_name = args.exp_name # 实验名称


    # 创建保存目录和日志
    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name) # 结果保存路径
    if not os.path.exists(save_folder):
        os.makedirs(save_folder) # 递归创建目录
        print("makedirs {}".format(save_folder)) # 打印创建信息

    # 初始化日志记录器
    save_log_path = os.path.join(save_folder,'train_tr{}_val{}.log'.format(num_list[0],num_list[1])) #日志文件路径
    logger = setup_logger(name='{}'.format(dataset_name),logfile=save_log_path) #创建日志记录器
    torch.cuda.empty_cache() # 清空GPU缓存

    logger.info(save_folder)

    # 加载数据与预处理
    data, gt = data_load_operate.load_data(data_set_name, data_set_path) # 加载数据和标签
    height, width, channels = data.shape # 数据维度
    gt_reshape = gt.reshape(-1) # 标签展平
    height, width, channels = data.shape # 数据归一化
    img = ImageStretching(data) # 类别数量
    class_count = max(np.unique(gt))  # 统计类别数量（标签最大值）

    # 训练/验证集划分
    flag_list = [1, 0]  # ratio or num 采样方式标志
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio] 训练集和验证集比例

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1) #定义交叉损失函数，忽略标签为-1的像素


    OA_ALL = []         # 存储每次实验的总体准确率 (Overall Accuracy)
    AA_ALL = []         # 存储每次实验的平均类别准确率 (Average Accuracy)
    KPP_ALL = []        # 存储每次实验的Kappa系数 (Kappa Coefficient)
    EACH_ACC_ALL = []   # 存储每次实验的每个类别的准确率 (Per-class Accuracy)
    Train_Time_ALL = [] # 存储每次实验的训练时间
    Test_Time_ALL = []  # 存储每次实验的测试时间

    CLASS_ACC = np.zeros([len(seed_list), class_count]) # 初始化类别准确率矩阵
    evaluator = Evaluator(num_class=class_count)  # 创建评估器对象

    for exp_idx,curr_seed in enumerate(seed_list): # 循环遍历随机种子列表
        setup_seed(curr_seed) # 把实验随机数锁死
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed)) # 创建专属文件夹，名字为第几次实验用哪个随机种子
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        # 如果文件夹不存在就新建一个
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        # 再实验文件夹里建一个vis文件夹 （存预测结果）
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))
        # 保定义保存路径
        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        # 数据划分，把数据划分成训练集，验证集，测试集
        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                       num_list,
                                                                                                       gt_reshape,
                                                                                                       class_count,
                                                                                                       flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data, height, width, gt_reshape, index)

        # build Model 新建一个ACmamba神经网络
        net = ACmamba(in_channels=channels, num_classes=class_count, hidden_dim=128) # 输入通道数，要分类的类别总数，网络中间层神经元个数
        logger.info(paras_dict) # 打印所有参数
        logger.info(net) # 打印网络结构

        x = transform(np.array(img))  # 把图像转换成张量
        x = x.unsqueeze(0).float().to(device) # 增加一个批次维度，并放到GPU上


        train_label = train_label.to(device) # 把训练标签放到GPU
        test_label = test_label.to(device)   # 测试标签
        val_label = val_label.to(device)     # 验证标签

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device) # 把神经网络也放到GPU上

        # 初始化记录列表
        train_loss_list = [100] # 记录训练损失
        train_acc_list = [0]    # 记录训练准确率
        val_loss_list = [100]   # 记录验证损失
        val_acc_list = [0]      # 记录验证准确率

        # 设置优化器 用Adam算法来调整网络参数
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        logger.info(optimizer)
        best_loss = 99999

        # 计算模型复杂度
        if record_computecost:
            net.eval()
            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]), )
            logger.info("para:{}\n,flops:{}".format(para, flops))
            # para 参数数量 flops计算量

        tic1 = time.perf_counter() # 记录开始时间
        best_val_acc = 0           # 初始化最佳验证准确率

        time_epoch = time.time()  # 记录epoch开始时间
        # 训练循环初始化
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0) # 增加批次维度
            train_acc_sum, trained_samples_counter = 0.0, 0 # 初始化训练统计量
            batch_counter, train_loss_sum = 0, 0

            loss_dict = {} # 存储损失值

            net.train() # 设置为训练模式

            if split_image: # 如果启用图像分割训练
                # 将图像垂直分割为两部分（有5像素重叠）
                x_part1 = x[:, :, :x.shape[2] // 2+5, :]
                y_part1 = y_train[:,:x.shape[2] // 2+5,:]
                x_part2 = x[:, :, x.shape[2] // 2 - 5: , :]
                y_part2 = y_train[:,x.shape[2] // 2 - 5:,:]

                # 第一部分前向传播
                y_pred_part1 = net(x_part1)
                ls1 = head_loss(loss_func,y_pred_part1, y_part1.long())
                optimizer.zero_grad()   # 清空梯度
                ls1.backward()  # 反向传播
                optimizer.step() # 更新参数
                torch.cuda.empty_cache() # 清理GPU缓存

                # 第二部分前向传播
                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func,y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))


            else:
                try:
                    # 尝试完整图像训练
                    y_pred = net(x)
                    ls = head_loss(loss_func,y_pred, y_train.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    logger.info('Iter:{}|loss:{}'.format(epoch, ls.detach().cpu().numpy()))
                except:
                    # 如果内存不足，自动切换到分割模式
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image=True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    y_pred_part1 = net(x_part1)
                    ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    y_pred_part2 = net(x_part2)
                    ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    logger.info(
                        'Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            torch.cuda.empty_cache()
            # evaluate stage
            net.eval() # 设置为评估模式
            with torch.no_grad(): # 禁用梯度计算
                evaluator.reset()  # 重置评估器
                # output_val = net(x)
                output_val = net(x) # 验证集前向传播
                y_val = val_label.unsqueeze(0)

                # 调整输出尺寸匹配标签
                seg_logits = resize(input=output_val,
                                    size=y_val.shape[1:],
                                    mode='bilinear',
                                    align_corners=True)

                # 计算预测结果
                predict = torch.argmax(seg_logits,dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np) # 特殊标签处理

                # 更新评估器
                evaluator.add_batch(np.expand_dims(Y_val_255,axis=0),predict)
                # 计算各项指标
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA,mAcc,Kappa,mIOU,IOU,Acc))
                # save weight
                # 记录最佳模型
                if OA>=best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    # torch.save(net,save_weight_path)
                    torch.save(net.state_dict(), save_weight_path) # 保存模型权重
                    # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                    # torch.save(net.state_dict(), save_epoch_weight_path)
                    # net.eval()
                    # with torch.no_grad():
                    #     output_val = net(x)
                    #
                    #     # 1. 保存测试集特征（用于论文主图）
                    #     test_seg_logits = resize(input=output_val,
                    #                              size=test_label.unsqueeze(0).shape[1:],
                    #                              mode='bilinear',
                    #                              align_corners=True)
                    #     test_features = test_seg_logits.detach().cpu().numpy()
                    #     np.save(f'{save_single_experiment_folder}/best_model_test_features_seed{curr_seed}.npy',
                    #             test_features)
                    #     np.save(f'{save_single_experiment_folder}/best_model_test_labels_seed{curr_seed}.npy',
                    #             test_label.cpu().numpy())
                    #
                    #     # 2. 保存全图特征（用于补充材料）
                    #     full_seg_logits = resize(input=output_val,
                    #                              size=gt.shape,
                    #                              mode='bilinear',
                    #                              align_corners=True)
                    #     full_features = full_seg_logits.detach().cpu().numpy()
                    #     np.save(f'{save_single_experiment_folder}/best_model_full_features_seed{curr_seed}.npy',
                    #             full_features)
                    #     np.save(f'{save_single_experiment_folder}/best_model_full_labels_seed{curr_seed}.npy', gt)
                    #
                    #     # print(f" 保存最佳模型特征:")
                    #     # print(f"  - 测试集: {test_features.shape}")
                    #     # print(f"  - 全图: {full_features.shape}")

                # 定期保存预测可视化
                if (epoch+1)%50==0:
                    save_single_predict_path = os.path.join(save_vis_folder,'predict_{}.png'.format(str(epoch+1)))
                    save_single_gt_path = os.path.join(save_vis_folder,'gt.png')
                    vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)

                # net.train()
            torch.cuda.empty_cache()
        time_epoch_end = time.time()  # 记录epoch开始时间
        train_time_total = time_epoch_end - time_epoch
        Train_Time_ALL.append(train_time_total) # 记录本次实验训练时间

        # net.eval()
        # with torch.no_grad():
        #     output_val = net(x)

            # 1. 测试集特征
            # test_seg_logits = resize(input=output_val,
            #                          size=test_label.unsqueeze(0).shape[1:],
            #                          mode='bilinear',
            #                          align_corners=True)
            # test_features = test_seg_logits.detach().cpu().numpy()
            # np.save(f'{save_single_experiment_folder}/final_model_test_features_seed{curr_seed}.npy', test_features)
            # np.save(f'{save_single_experiment_folder}/final_model_test_labels_seed{curr_seed}.npy',
            #         test_label.cpu().numpy())

            # 2. 全图特征
            # full_seg_logits = resize(input=output_val,
            #                          size=gt.shape,
            #                          mode='bilinear',
            #                          align_corners=True)
            # full_features = full_seg_logits.detach().cpu().numpy()
            # np.save(f'{save_single_experiment_folder}/final_model_full_features_seed{curr_seed}.npy', full_features)
            # np.save(f'{save_single_experiment_folder}/final_model_full_labels_seed{curr_seed}.npy', gt)

            # print(f" 保存最终模型特征:")
            # print(f"  - 测试集: {test_features.shape}")
            # print(f"  - 全图: {full_features.shape}")



        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        # 加载最佳模型
        load_weight_path = save_weight_path
        net.update_params = None


        # best_net = copy.deepcopy(net)
        best_net = ACmamba(in_channels=channels, num_classes=class_count, hidden_dim=128)

        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()

        best_net.eval()
        with torch.no_grad():
            output_test = best_net(x)

            # 1. 测试集特征
            test_seg_logits = resize(input=output_test,
                                     size=test_label.unsqueeze(0).shape[1:],
                                     mode='bilinear',
                                     align_corners=True)
            test_features = test_seg_logits.detach().cpu().numpy()
            np.save(f'{save_single_experiment_folder}/test_best_test_features_seed{curr_seed}.npy', test_features)
            np.save(f'{save_single_experiment_folder}/test_best_test_labels_seed{curr_seed}.npy',
                    test_label.cpu().numpy())

            # 2. 全图特征
            full_seg_logits = resize(input=output_test,
                                     size=gt.shape,
                                     mode='bilinear',
                                     align_corners=True)
            full_features = full_seg_logits.detach().cpu().numpy()
            np.save(f'{save_single_experiment_folder}/test_best_full_features_seed{curr_seed}.npy', full_features)
            np.save(f'{save_single_experiment_folder}/test_best_full_labels_seed{curr_seed}.npy', gt)

            # print(f" 保存测试最佳模型特征:")
            # print(f"  - 测试集: {test_features.shape}")
            # print(f"  - 全图: {full_features.shape}")


        time_test = time.time()

        test_evaluator = Evaluator(num_class=class_count)
        # 测试集评估
        with torch.no_grad():
            test_evaluator.reset()
            output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test,
                                size=y_test.shape[1:],
                                mode='bilinear',
                                align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)

            # 处理测试结果
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            # 记录结果
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                    Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)# 保存可视化结果
        time_test_end = time.time()
        test_time_total = time_test_end - time_test
        Test_Time_ALL.append(test_time_total)  # 记录本次实验训练时间


        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        # 汇总多次实验的结果
        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    # 计算均值和标准差
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', list(OA_ALL))
    logger.info('List of AA:', list(AA_ALL))
    logger.info('List of KPP:', list(KPP_ALL))
    logger.info('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
          np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

    # Output infors
    # 保存结果到文件
    mean_result_path = os.path.join(save_folder,'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(
        np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(
        np.round(np.mean(Test_Time_ALL)*1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL)*1000, decimals=3))
    f.write(str_results) # 写入格式化后的结果字符串
    f.close()

    del net
