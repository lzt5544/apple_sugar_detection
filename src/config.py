import os

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# 模型路径
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 结果路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# 模型参数
MODEL_NAME = 'resnet50'  # 可选: 'resnet50', 'custom'
PRETRAINED = True
FREEZE_LAYERS = True
IMG_SIZE = 224

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
NUM_WORKERS = 4  # 数据加载器工作进程数

# 回调参数
SAVE_INTERVAL = 10
SAVE_BEST_ONLY = True
EARLY_STOPPING_PATIENCE = 10
TENSORBOARD_LOGGING = True

# 确保目录存在
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

if __name__ == '__main__':
    ensure_dirs()
    print(f'项目根目录: {PROJECT_ROOT}')
    print(f'数据目录: {DATA_DIR}')
    print(f'模型目录: {MODELS_DIR}')
    print(f'结果目录: {RESULTS_DIR}')
    print('配置加载完成!')