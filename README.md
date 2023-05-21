# 临时数据集
cifar10 & cifar100：
https://box.nju.edu.cn/d/167ca1e2252c4d078bf7/

# LibContinual
A framework of Continual Learning


# 代码结构
1. data模块：'./core/data' 负责dataset的读取逻辑，关于datasets所需要的transform暂时没有写好，大家在复现各自方法的时候，需要什么transform可以直接修改./core/data/dataloader.py，后续会逐渐完善. <br>

2. bakcbone模块：'./core/model/backbone' 负责backbone模型文件的定义(不包含fc)，这里我是参考PyCIL(https://github.com/G-U-N/PyCIL).   建议大家在复现各自方法之前，先检查一下与论文代码中的模型结构是否一致。   <br>

3. buffer模块： './core/model/buffer' 负责训练过程中buffer的管理以及更新。 目前只实现了LinearBuffer, 在每个任务开始前会把buffer样本与新样本拼接在一起.  buffer的更新策略，目前只支持了随机更新.  其他类型的Buffer以及更新策略后续会逐渐完善.  此外，Buffer的更新函数 def update的参数，大家在实现的时候可以先根据自己的需要随意设置，后续会考虑整合.  <br>

4. logger模块：'./core/utils/logger.py' 负责训练过程中的日志打印。 此处选择直接hack 系统输出，因此大家在训练过程中不需要显示的调用logger.info等接口，  直接正常的print想要的信息，logger模块会自动的保存在日志文件中.  <br>

5. trainer模块：'./core/trainer.py' 负责整个实验的流程控制。 大家在复现各自方法的时候，如果流程上有暂时支持不了的，可以直接修改trainer.py来满足，并且可以反馈给我，后续我会对流程做进一步的完善.  <br>

6. config模块：'./config/', 负责整个训练过程的参数配置。
   入口：run_trainer.py里的line：15填入各自方法对应的yaml配置文件路径。 为每个方法在./config/路径下新建一个*.yaml。 配置文件里面需要写入以下参数： <br>
   a/  includes:  仿照finetune.yaml照抄，用来填充一些默认的参数。   *.yaml里的参数会覆盖掉config/headers/里的参数  <br>
   b/  data_root: 所使用的数据集路径。数据集的摆放格式如下：
         data_root:  <br>
         | ---train  <br>
         | ------class1   <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class2  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class3  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ---test  <br>
         | ------class1  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class2  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class3  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>

   c/ save_path: log以及checkpoints存放路径，log文件存放在 save_path/xxx.log,  checkpoint保存功能还未完成.  <br>

   d/ init_cls_num, inc_cls_num, task_num: 第一个任务的类别个数、后续每个任务的类别个数、任务总数. 类别顺序是随机生成的 <br>

   e/ init_epoch, epoch:  第一个任务的训练epoch数、后续每个任务的训练epoch数，没设置init_epoch的情况下init_epoch = epoch  <br>

   f/ backbone:  参考finetune.yaml, 一般指明name即可， 其中args:datasets 是代码遗留问题，暂时先照抄，后续会修改掉.   <br>

   g/ optimizer, lr_scheduler:  可以仿照大家平常使用pytorch自带的optimizer与scheduler, 将名字与相应参数，仿照finetune.yaml的形式填入即可.   <br>

   h/ buffer:  与训练过程中使用的buffer相关，目前buffer的使用只支持将旧样本与新样本拼接在一起。buffer_size, batch_size, strategy： 旧样本数量，batch_size在linearBuffer下没用，strategy更新策略。

   i: classifier: name:对应各自实现的方法名，其他参数看各自需要什么，直接在里面加


# 复现需要做的事
1. 选定好一个方法后，在./config路径下新增一个.yaml文件用来满足需要的参数设置. 在./model/replay 或者 ./model/下新增一个.py文件用来实现训练算法.   <br>

2. 对于.py文件，需要实现几个函数: <br>
  def __init__():  用来初始化各自算法需要的对象
  def observe(self, data):  训练过程中，面对到来的一个batch的样本完成训练的损失计算以及参数更新，返回pred, acc, loss:  预测结果，准确率，损失    <br>
  def inference(self, data):   推理过程中，面对到来的一个batch的样本，完成预测，返回pred, acc   <br>
  def before_task() / after_task():  可选，如果算法在每个任务开始前后有额外的操作，在这两个函数内完成   <br>

3. 训练过程中需要不同的buffer以及更新策略，可以自己在'./cire/model/buffer'下仿照LinearBuffer新增文件，并反馈给我.


