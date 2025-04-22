import argparse
import os
import random
import shutil
import yaml
from glob import glob
from sklearn.model_selection import train_test_split

def parse():
    parser = argparse.ArgumentParser(
        description="Preprocess data directories and generate train/val/test splits."
    )
    parser.add_argument(
        "--data_dirs", type=str, required=True,
        help="Comma-separated list of directories containing data. Example: 'data/dir1,data/dir2'"
    )
    parser.add_argument(
        "--n_splits", type=int, default=10,
        help="Number of splits for cross-validation. If n_splits > 1, KFold will be generated. (default: 1)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.6,
        help="Ratio for training set. (default: 0.6)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Ratio for validation set. (default: 0.2)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="Ratio for testing set. (default: 0.2)"
    )
    parser.add_argument(
        "--data_count", type=int, default=None,
        help="If set, randomly select this many total samples from the dataset before splitting."
    )
    return parser.parse_args()

def create_random_splits(data_dir, n_splits=5, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, data_count=None):
    """
    단일 데이터 경로에서 이미지와 라벨 데이터셋을 N개의 랜덤 train/val/test로 분할하고 관련 파일을 생성합니다.
    
    Args:
        data_dir (str): 데이터 디렉토리 경로 (images, labels 폴더와 data.yaml 파일을 포함)
        n_splits (int): 생성할 데이터셋 분할 수
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
    """
    # 비율 합이 1인지 확인
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "비율의 합은 1이어야 합니다"
    
    # 경로 설정
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    yaml_path = os.path.join(data_dir, 'data.yaml')
    
    # 디렉토리 존재 확인
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"경고: {data_dir}에 images 또는 labels 폴더가 없습니다. 건너뜁니다.")
        return
        
    # 이미지 파일 목록 가져오기 (확장자는 필요에 따라 수정)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_dir, ext)))
    
    # 이미지 파일명만 추출 (확장자 포함)
    image_filenames = [os.path.basename(f) for f in image_files]
    
    # 해당 이미지 파일에 대응하는 라벨 파일이 있는지 확인
    valid_samples = []
    for img_filename in image_filenames:
        # 이미지 파일명에서 확장자를 제거하고 라벨 파일 확장자 추가
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        
        # 라벨 파일이 존재하면 유효한 샘플로 간주
        if os.path.exists(label_path):
            valid_samples.append((img_filename, base_name + '.txt'))
    
    if not valid_samples:
        print(f"오류: {data_dir}에서 유효한 샘플을 찾을 수 없습니다.")
        return
    
    # data.yaml 파일 불러오기
    try:
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            nc = data_yaml.get('nc', 0)
            names = data_yaml.get('names', [])
    except FileNotFoundError:
        print(f"경고: {yaml_path} 파일을 찾을 수 없습니다. nc와 names는 기본값으로 설정됩니다.")
        nc = 0
        names = []
    
    print(f"{data_dir}: {len(valid_samples)}개의 유효한 샘플을 찾았습니다.")
    
    # 유효 샘플 리스트(valid_samples)를 만든 뒤, 여기서 개수 제한
    if data_count is not None and len(valid_samples) > data_count:
        valid_samples = random.sample(valid_samples, data_count)

    print(f"{data_dir}: 최종 사용 샘플 {len(valid_samples)}개로 분할을 시작합니다.")
    
    # N개의 랜덤 분할 생성
    for iter_idx in range(1, n_splits + 1):
        # 데이터 세트 분할
        random.shuffle(valid_samples)
        
        # 먼저 train과 temp(val+test) 분할
        train_samples, temp_samples = train_test_split(
            valid_samples, 
            train_size=train_ratio, 
            test_size=val_ratio + test_ratio, 
            random_state=42 + iter_idx  # 각 반복마다 다른 시드 사용
        )
        
        # temp를 val과 test로 분할
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples, 
            train_size=val_ratio_adjusted, 
            test_size=1 - val_ratio_adjusted,
            random_state=42 + iter_idx
        )
        
        # 텍스트 파일 생성
        train_txt_path = os.path.join(data_dir, f'train_iter_{iter_idx:02d}.txt')
        val_txt_path = os.path.join(data_dir, f'val_iter_{iter_idx:02d}.txt')
        test_txt_path = os.path.join(data_dir, f'test_iter_{iter_idx:02d}.txt')
        
        # 학습 데이터 파일 쓰기
        with open(train_txt_path, 'w') as f:
            for img_file, _ in train_samples:
                img_path = os.path.join(data_dir, 'images', img_file)  # 상대 경로 사용
                f.write(f"{img_path}\n")
        
        # 검증 데이터 파일 쓰기
        with open(val_txt_path, 'w') as f:
            for img_file, _ in val_samples:
                img_path = os.path.join(data_dir, 'images', img_file)  # 상대 경로 사용
                f.write(f"{img_path}\n")
        
        # 테스트 데이터 파일 쓰기
        with open(test_txt_path, 'w') as f:
            for img_file, _ in test_samples:
                img_path = os.path.join(data_dir, 'images', img_file)  # 상대 경로 사용
                f.write(f"{img_path}\n")
        
        # YAML 파일 생성
        output_yaml_path = os.path.join(data_dir, f'data_iter_{iter_idx:02d}.yaml')
        yaml_content = {
            'path': data_dir,
            'train': f'train_iter_{iter_idx:02d}.txt',  # 상대 경로 사용
            'val': f'val_iter_{iter_idx:02d}.txt',      # 상대 경로 사용
            'test': f'test_iter_{iter_idx:02d}.txt',    # 상대 경로 사용
            'nc': nc,
            'names': names
        }
        
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"{data_dir} - Iteration {iter_idx} 완료: {len(train_samples)} 학습, {len(val_samples)} 검증, {len(test_samples)} 테스트")


def main(args):
    # data_dirs는 쉼표로 구분된 문자열을 리스트로 변환합니다.
    data_dirs = [d.strip() for d in args.data_dirs.split(",")]
    n_splits = args.n_splits
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    data_count = args.data_count

    print("입력된 데이터 디렉터리:", data_dirs)
    print("n_splits:", n_splits)
    print("학습 비율:", train_ratio)
    print("검증 비율:", val_ratio)
    print("테스트 비율:", test_ratio)
    print("사용할 데이터 수:", data_count if data_count is not None else 'all')

    # 데이터 전처리 실행
    for data_dir in data_dirs:
        print(f"\n처리 중: {data_dir}")
        create_random_splits(data_dir, n_splits, train_ratio, val_ratio, test_ratio, data_count)

    # 추가적으로 생성된 데이터셋을 파일로 저장하거나, 후속 처리를 진행할 수 있습니다.
    print("전처리 완료.")

if __name__ == '__main__':
    args = parse()
    main(args)