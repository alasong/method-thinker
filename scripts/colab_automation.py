#!/usr/bin/env python
"""Google Colab 自动化训练脚本

使用papermill参数化执行notebook，支持：
- Google Drive API 自动上传数据/下载结果
- 参数化训练配置
- 自动触发和结果收集
- 定时训练支持（通过Google Cloud Scheduler）

用法示例:
    # 基础用法 - 本地执行notebook
    python scripts/colab_automation.py execute \
        --notebook notebooks/train_automated.ipynb \
        --config configs/colab_automation.yaml

    # 上传数据到Drive并准备训练
    python scripts/colab_automation.py prepare \
        --config configs/colab_automation.yaml

    # 执行训练并下载结果
    python scripts/colab_automation.py run \
        --config configs/colab_automation.yaml \
        --download-results

    # 使用预设训练配置
    python scripts/colab_automation.py run \
        --preset quick \
        --download-results

    # 定时训练（需要Google Cloud Scheduler）
    python scripts/colab_automation.py schedule \
        --config configs/colab_automation.yaml \
        --cron "0 9 * * 1"  # 每周一9点
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 配置管理
# ============================================================================

class AutomationConfig:
    """自动化配置管理"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            # 尝试默认配置路径
            default_path = "configs/colab_automation.yaml"
            if os.path.exists(default_path):
                self.config_path = default_path
            else:
                print(f"警告: 配置文件不存在，使用默认配置")
                return self._get_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'google_cloud': {
                'project_id': None,
                'credentials_path': None,
                'region': 'us-central1'
            },
            'google_drive': {
                'enabled': True,
                'base_folder': 'method-thinker-models',
                'data_folder': 'training-data',
                'results_folder': 'training-results',
                'notebooks_folder': 'notebooks'
            },
            'training': {
                'base_model': 'Qwen/Qwen2.5-Math-1.5B',
                'epochs': 3,
                'batch_size': 4,
                'learning_rate': 5e-5,
                'max_length': 2048,
                'lora_r': 16,
                'lora_alpha': 32,
                'hf_mirror': 'https://hf-mirror.com'
            },
            'presets': {
                'quick': {
                    'epochs': 1,
                    'max_samples': 100,
                    'description': '快速验证训练'
                },
                'standard': {
                    'epochs': 3,
                    'max_samples': 1000,
                    'description': '标准训练'
                },
                'full': {
                    'epochs': 5,
                    'max_samples': 5000,
                    'description': '完整训练'
                }
            },
            'paths': {
                'notebook_template': 'notebooks/train_automated.ipynb',
                'output_notebook': 'outputs/executed_notebooks/',
                'results_dir': 'outputs/colab_results/',
                'local_data_dir': 'data/train_data/',
                'local_kb_dir': 'data/methodology_kb/'
            },
            'scheduling': {
                'enabled': False,
                'scheduler_type': 'local',  # local, gcloud
                'cron': None
            }
        }

    def get_preset(self, preset_name: str) -> Dict:
        """获取预设配置"""
        presets = self.config.get('presets', {})
        if preset_name not in presets:
            print(f"警告: 预设 '{preset_name}' 不存在，使用默认配置")
            return {}
        return presets[preset_name]

    def get_training_params(self) -> Dict:
        """获取训练参数"""
        return self.config.get('training', {})

    def get_drive_config(self) -> Dict:
        """获取Drive配置"""
        return self.config.get('google_drive', {})


# ============================================================================
# Papermill Notebook 执行器
# ============================================================================

class NotebookExecutor:
    """参数化Notebook执行器"""

    def __init__(self, config: AutomationConfig):
        self.config = config

    def inject_parameters(
        self,
        template_path: str,
        output_path: str,
        parameters: Dict
    ) -> str:
        """注入参数到notebook

        Args:
            template_path: 模板notebook路径
            output_path: 输出notebook路径
            parameters: 要注入的参数

        Returns:
            输出notebook路径
        """
        try:
            import papermill as pm
        except ImportError:
            print("错误: papermill未安装")
            print("请运行: pip install papermill")
            raise

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 执行notebook并注入参数
        print(f"执行notebook: {template_path}")
        print(f"输出位置: {output_path}")
        print(f"注入参数: {json.dumps(parameters, indent=2)}")

        pm.execute_notebook(
            template_path,
            output_path,
            parameters=parameters,
            kernel_name='python3',
            log_output=True,
            progress_bar=True
        )

        print(f"Notebook执行完成: {output_path}")
        return output_path

    def extract_results(self, notebook_path: str) -> Dict:
        """从执行的notebook中提取结果

        Args:
            notebook_path: 执行后的notebook路径

        Returns:
            结果字典
        """
        try:
            import papermill as pm
        except ImportError:
            raise ImportError("papermill未安装")

        # 读取notebook并提取输出
        nb = pm.read_notebook(notebook_path)

        results = {}
        for cell in nb.cells:
            if cell.cell_type == 'code' and cell.outputs:
                # 查找特定输出标签
                for output in cell.outputs:
                    if output.output_type == 'execute_result':
                        if 'data' in output and 'text/plain' in output.data:
                            text = output.data['text/plain']
                            # 提取JSON结果
                            if 'training_report' in text:
                                try:
                                    # 尝试解析JSON
                                    results['training_report'] = self._parse_output(text)
                                except:
                                    results['raw_output'] = text

        return results

    def _parse_output(self, text: str) -> Dict:
        """解析输出文本"""
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {'raw': text}


# ============================================================================
# Google Drive 同步器
# ============================================================================

class DriveSyncer:
    """Google Drive 文件同步器"""

    def __init__(self, config: AutomationConfig):
        self.config = config
        self.drive_config = config.get_drive_config()
        self.service = None
        self._initialized = False

    def _initialize(self):
        """初始化Drive API"""
        if self._initialized:
            return

        drive_enabled = self.drive_config.get('enabled', False)
        if not drive_enabled:
            print("Google Drive同步已禁用")
            return

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            print("错误: Google API客户端未安装")
            print("请运行: pip install google-api-python-client google-auth")
            return

        gc_config = self.config.config.get('google_cloud', {})
        credentials_path = gc_config.get('credentials_path')

        if credentials_path and os.path.exists(credentials_path):
            # 使用服务账号认证
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            self.service = build('drive', 'v3', credentials=credentials)
            self._initialized = True
            print("Google Drive API已初始化（服务账号）")
        else:
            print("警告: 未找到Google Cloud凭据文件")
            print("请设置 google_cloud.credentials_path 配置项")

    def _get_or_create_folder(self, folder_name: str, parent_id: str = None) -> str:
        """获取或创建文件夹

        Args:
            folder_name: 文件夹名称
            parent_id: 父文件夹ID

        Returns:
            文件夹ID
        """
        if not self.service:
            return None

        # 搜索文件夹
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        query += " and trashed=false"

        results = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])
        if files:
            return files[0]['id']

        # 创建文件夹
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]

        file = self.service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()

        return file.get('id')

    def upload_file(
        self,
        local_path: str,
        remote_name: str = None,
        folder_type: str = 'data'
    ) -> Optional[str]:
        """上传文件到Drive

        Args:
            local_path: 本地文件路径
            remote_name: 远程文件名（可选）
            folder_type: 目标文件夹类型 (data/results/notebooks)

        Returns:
            文件ID
        """
        self._initialize()

        if not self.service:
            print("Drive服务未初始化，跳过上传")
            return None

        if not os.path.exists(local_path):
            print(f"文件不存在: {local_path}")
            return None

        # 获取目标文件夹
        folder_mapping = {
            'data': self.drive_config.get('data_folder', 'training-data'),
            'results': self.drive_config.get('results_folder', 'training-results'),
            'notebooks': self.drive_config.get('notebooks_folder', 'notebooks')
        }

        target_folder = folder_mapping.get(folder_type, 'training-data')
        base_folder = self.drive_config.get('base_folder', 'method-thinker-models')

        # 获取文件夹ID
        base_id = self._get_or_create_folder(base_folder)
        target_id = self._get_or_create_folder(target_folder, base_id)

        # 文件名
        if not remote_name:
            remote_name = os.path.basename(local_path)

        # 上传
        from googleapiclient.http import MediaFileUpload

        file_metadata = {
            'name': remote_name,
            'parents': [target_id]
        }

        media = MediaFileUpload(
            local_path,
            mimetype='application/octet-stream',
            resumable=True
        )

        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()

        file_id = file.get('id')
        print(f"文件已上传: {remote_name} (ID: {file_id})")
        return file_id

    def download_file(
        self,
        file_id: str,
        local_path: str
    ) -> bool:
        """从Drive下载文件

        Args:
            file_id: Drive文件ID
            local_path: 本地保存路径

        Returns:
            是否成功
        """
        self._initialize()

        if not self.service:
            return False

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        request = self.service.files().get_media(fileId=file_id)

        with open(local_path, 'wb') as f:
            f.write(request.execute())

        print(f"文件已下载: {local_path}")
        return True

    def list_files(self, folder_type: str = 'results') -> List[Dict]:
        """列出文件夹中的文件

        Args:
            folder_type: 文件夹类型

        Returns:
            文件列表
        """
        self._initialize()

        if not self.service:
            return []

        folder_mapping = {
            'data': self.drive_config.get('data_folder', 'training-data'),
            'results': self.drive_config.get('results_folder', 'training-results'),
            'notebooks': self.drive_config.get('notebooks_folder', 'notebooks')
        }

        target_folder = folder_mapping.get(folder_type, 'training-results')
        base_folder = self.drive_config.get('base_folder', 'method-thinker-models')

        base_id = self._get_or_create_folder(base_folder)
        target_id = self._get_or_create_folder(target_folder, base_id)

        results = self.service.files().list(
            q=f"'{target_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id, name, createdTime, size)',
            orderBy='createdTime desc'
        ).execute()

        return results.get('files', [])

    def sync_training_data(self, local_dir: str) -> Dict[str, str]:
        """同步训练数据到Drive

        Args:
            local_dir: 本地数据目录

        Returns:
            上传文件ID映射
        """
        self._initialize()

        uploaded = {}

        if not os.path.exists(local_dir):
            print(f"数据目录不存在: {local_dir}")
            return uploaded

        for filename in ['train.json', 'val.json', 'test.json']:
            filepath = os.path.join(local_dir, filename)
            if os.path.exists(filepath):
                file_id = self.upload_file(filepath, filename, 'data')
                if file_id:
                    uploaded[filename] = file_id

        # 上传知识库
        kb_dir = self.config.config.get('paths', {}).get('local_kb_dir', 'data/methodology_kb/')
        kb_path = os.path.join(kb_dir, 'v0/math_methods.yaml')
        if os.path.exists(kb_path):
            file_id = self.upload_file(kb_path, 'math_methods.yaml', 'data')
            if file_id:
                uploaded['kb'] = file_id

        return uploaded

    def download_latest_results(self, local_dir: str) -> Optional[str]:
        """下载最新的训练结果

        Args:
            local_dir: 本地保存目录

        Returns:
            下载的文件路径
        """
        self._initialize()

        files = self.list_files('results')

        if not files:
            print("没有找到训练结果")
            return None

        # 获取最新文件
        latest = files[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.zip"
        local_path = os.path.join(local_dir, filename)

        self.download_file(latest['id'], local_path)

        return local_path


# ============================================================================
# Colab 执行器（本地模拟）
# ============================================================================

class ColabExecutor:
    """Colab执行器（本地papermill执行）"""

    def __init__(self, config: AutomationConfig):
        self.config = config
        self.notebook_executor = NotebookExecutor(config)
        self.drive_syncer = DriveSyncer(config)

    def prepare(self, args: argparse.Namespace) -> Dict:
        """准备训练环境

        Args:
            args: 命令行参数

        Returns:
            准备结果
        """
        results = {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

        # 1. 检查本地数据
        paths_config = self.config.config.get('paths', {})
        local_data_dir = paths_config.get('local_data_dir', 'data/train_data/')

        data_files = []
        for filename in ['train.json', 'val.json', 'test.json']:
            filepath = os.path.join(local_data_dir, filename)
            if os.path.exists(filepath):
                data_files.append(filepath)

        results['local_data'] = data_files

        # 2. 同步到Drive
        if args.sync_drive or self.config.get_drive_config().get('enabled'):
            print("\n同步数据到Google Drive...")
            uploaded = self.drive_syncer.sync_training_data(local_data_dir)
            results['drive_uploaded'] = uploaded

        # 3. 检查notebook模板
        notebook_template = paths_config.get('notebook_template', 'notebooks/train_automated.ipynb')
        if os.path.exists(notebook_template):
            results['notebook_template'] = notebook_template
        else:
            results['status'] = 'error'
            results['error'] = f"Notebook模板不存在: {notebook_template}"

        # 4. 检查依赖
        try:
            import papermill
            import transformers
            import peft
            results['dependencies'] = 'ok'
        except ImportError as e:
            results['status'] = 'error'
            results['error'] = f"缺少依赖: {e}"

        print("\n准备完成!")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return results

    def execute(self, args: argparse.Namespace) -> Dict:
        """执行训练notebook

        Args:
            args: 命令行参数

        Returns:
            执行结果
        """
        # 1. 获取训练参数
        training_params = self.config.get_training_params()

        # 2. 应用预设（如果有）
        if args.preset:
            preset = self.config.get_preset(args.preset)
            training_params.update(preset)

        # 3. 应用命令行参数覆盖
        if args.base_model:
            training_params['base_model'] = args.base_model
        if args.epochs:
            training_params['epochs'] = args.epochs
        if args.batch_size:
            training_params['batch_size'] = args.batch_size
        if args.learning_rate:
            training_params['learning_rate'] = args.learning_rate
        if args.max_length:
            training_params['max_length'] = args.max_length

        # 4. 构建papermill参数
        parameters = {
            'BASE_MODEL': training_params.get('base_model', 'Qwen/Qwen2.5-Math-1.5B'),
            'NUM_EPOCHS': training_params.get('epochs', 3),
            'BATCH_SIZE': training_params.get('batch_size', 4),
            'LEARNING_RATE': training_params.get('learning_rate', 5e-5),
            'MAX_LENGTH': training_params.get('max_length', 2048),
            'LORA_R': training_params.get('lora_r', 16),
            'LORA_ALPHA': training_params.get('lora_alpha', 32),
            'HF_MIRROR': training_params.get('hf_mirror', 'https://hf-mirror.com'),
            'MAX_SAMPLES': training_params.get('max_samples', 1000),
            'OUTPUT_DIR': 'outputs/colab_training',
            'RUN_TIMESTAMP': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # 5. 获取notebook路径
        paths_config = self.config.config.get('paths', {})
        template_path = args.notebook or paths_config.get('notebook_template', 'notebooks/train_automated.ipynb')

        if not os.path.exists(template_path):
            print(f"错误: Notebook不存在: {template_path}")
            return {'status': 'error', 'error': 'notebook_not_found'}

        # 6. 生成输出路径
        output_dir = paths_config.get('output_notebook', 'outputs/executed_notebooks/')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'train_executed_{timestamp}.ipynb')

        # 7. 执行notebook
        print("\n开始执行训练...")
        print(f"参数: {json.dumps(parameters, indent=2)}")

        try:
            executed_path = self.notebook_executor.inject_parameters(
                template_path,
                output_path,
                parameters
            )

            # 8. 提取结果
            results = self.notebook_executor.extract_results(executed_path)
            results['executed_notebook'] = executed_path
            results['parameters'] = parameters
            results['status'] = 'success'

            print("\n执行完成!")
            print(json.dumps(results, indent=2, ensure_ascii=False))

            return results

        except Exception as e:
            print(f"执行失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def run(self, args: argparse.Namespace) -> Dict:
        """完整运行流程

        Args:
            args: 命令行参数

        Returns:
            运行结果
        """
        # 1. 准备阶段
        prepare_result = self.prepare(args)
        if prepare_result.get('status') == 'error':
            return prepare_result

        # 2. 执行阶段
        execute_result = self.execute(args)
        if execute_result.get('status') == 'error':
            return execute_result

        # 3. 下载结果
        if args.download_results:
            print("\n下载训练结果...")
            results_dir = self.config.config.get('paths', {}).get('results_dir', 'outputs/colab_results/')
            downloaded = self.drive_syncer.download_latest_results(results_dir)
            execute_result['downloaded_results'] = downloaded

        return execute_result

    def schedule(self, args: argparse.Namespace) -> Dict:
        """设置定时训练

        Args:
            args: 命令行参数

        Returns:
            设置结果
        """
        scheduler_type = self.config.config.get('scheduling', {}).get('scheduler_type', 'local')

        if scheduler_type == 'gcloud':
            return self._setup_gcloud_scheduler(args)
        else:
            return self._setup_local_scheduler(args)

    def _setup_gcloud_scheduler(self, args: argparse.Namespace) -> Dict:
        """设置Google Cloud Scheduler"""
        gc_config = self.config.config.get('google_cloud', {})
        project_id = gc_config.get('project_id')
        region = gc_config.get('region', 'us-central1')

        if not project_id:
            return {'status': 'error', 'error': '未设置Google Cloud项目ID'}

        # 使用gcloud命令创建调度任务
        cron = args.cron or self.config.config.get('scheduling', {}).get('cron')
        if not cron:
            return {'status': 'error', 'error': '未设置cron表达式'}

        # 创建Cloud Scheduler job
        job_name = f"method-thinker-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 构建命令
        cmd = [
            'gcloud', 'scheduler', 'jobs', 'create', 'http', job_name,
            '--project', project_id,
            '--location', region,
            '--schedule', cron,
            '--uri', 'https://colab.research.google.com/github/your-repo/method-thinker/blob/main/notebooks/train_automated.ipynb',
            '--http-method', 'GET',
            '--time-zone', 'Asia/Shanghai'
        ]

        try:
            subprocess.run(cmd, check=True)
            return {
                'status': 'success',
                'job_name': job_name,
                'cron': cron,
                'message': 'Google Cloud Scheduler任务已创建'
            }
        except subprocess.CalledProcessError as e:
            return {'status': 'error', 'error': str(e)}

    def _setup_local_scheduler(self, args: argparse.Namespace) -> Dict:
        """设置本地调度（使用cron或Python schedule库）"""
        cron = args.cron or self.config.config.get('scheduling', {}).get('cron')

        if not cron:
            return {'status': 'error', 'error': '未设置cron表达式'}

        # 解析cron表达式
        # 格式: minute hour day month weekday
        parts = cron.split()
        if len(parts) != 5:
            return {'status': 'error', 'error': 'cron格式错误'}

        # 创建调度脚本
        script_dir = 'scripts/scheduled/'
        os.makedirs(script_dir, exist_ok=True)

        script_path = os.path.join(script_dir, f'scheduled_training_{datetime.now().strftime("%Y%m%d%H%M%S")}.sh')

        # 写入脚本
        with open(script_path, 'w') as f:
            f.write(f'''#!/bin/bash
# MethodThinker 自动训练脚本
# 创建时间: {datetime.now().isoformat()}
# Cron: {cron}

cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
python scripts/colab_automation.py run --config {self.config.config_path}
''')

        os.chmod(script_path, 0o755)

        # 显示如何添加到crontab
        print("\n本地调度脚本已创建")
        print(f"脚本路径: {script_path}")
        print(f"\n添加到crontab:")
        print(f"  crontab -e")
        print(f"  添加行: {cron} {script_path}")

        return {
            'status': 'success',
            'script_path': script_path,
            'cron': cron,
            'message': '请手动添加到crontab'
        }


# ============================================================================
# 主命令
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MethodThinker Colab自动化训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  prepare    准备训练环境（上传数据、检查依赖）
  execute    执行参数化notebook
  run        完整运行流程（准备 + 执行 + 下载结果）
  schedule   设置定时训练

预设配置:
  quick      快速验证训练 (1 epoch, 100 samples)
  standard   标准训练 (3 epochs, 1000 samples)
  full       完整训练 (5 epochs, 5000 samples)

示例:
  # 快速训练
  %(prog)s run --preset quick --download-results

  # 自定义参数
  %(prog)s execute --base-model Qwen/Qwen2.5-Math-1.5B --epochs 5

  # 定时训练
  %(prog)s schedule --cron "0 9 * * 1"  # 每周一9点
"""
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # prepare 命令
    prepare_parser = subparsers.add_parser('prepare', help='准备训练环境')
    prepare_parser.add_argument('--config', type=str, default='configs/colab_automation.yaml', help='配置文件路径')
    prepare_parser.add_argument('--sync-drive', action='store_true', help='同步数据到Drive')

    # execute 命令
    execute_parser = subparsers.add_parser('execute', help='执行参数化notebook')
    execute_parser.add_argument('--config', type=str, default='configs/colab_automation.yaml', help='配置文件路径')
    execute_parser.add_argument('--notebook', type=str, help='Notebook模板路径')
    execute_parser.add_argument('--preset', type=str, choices=['quick', 'standard', 'full'], help='使用预设配置')
    execute_parser.add_argument('--base-model', type=str, help='基座模型')
    execute_parser.add_argument('--epochs', type=int, help='训练轮数')
    execute_parser.add_argument('--batch-size', type=int, help='批大小')
    execute_parser.add_argument('--learning-rate', type=float, help='学习率')
    execute_parser.add_argument('--max-length', type=int, help='最大序列长度')

    # run 命令
    run_parser = subparsers.add_parser('run', help='完整运行流程')
    run_parser.add_argument('--config', type=str, default='configs/colab_automation.yaml', help='配置文件路径')
    run_parser.add_argument('--preset', type=str, choices=['quick', 'standard', 'full'], help='使用预设配置')
    run_parser.add_argument('--sync-drive', action='store_true', help='同步数据到Drive')
    run_parser.add_argument('--download-results', action='store_true', help='下载训练结果')
    run_parser.add_argument('--base-model', type=str, help='基座模型')
    run_parser.add_argument('--epochs', type=int, help='训练轮数')

    # schedule 命令
    schedule_parser = subparsers.add_parser('schedule', help='设置定时训练')
    schedule_parser.add_argument('--config', type=str, default='configs/colab_automation.yaml', help='配置文件路径')
    schedule_parser.add_argument('--cron', type=str, help='Cron表达式 (如: "0 9 * * 1" 表示每周一9点)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化配置和执行器
    config = AutomationConfig(args.config)
    executor = ColabExecutor(config)

    # 执行命令
    if args.command == 'prepare':
        result = executor.prepare(args)
    elif args.command == 'execute':
        result = executor.execute(args)
    elif args.command == 'run':
        result = executor.run(args)
    elif args.command == 'schedule':
        result = executor.schedule(args)
    else:
        parser.print_help()
        return

    # 保存结果
    if result.get('status') == 'success':
        results_dir = config.config.get('paths', {}).get('results_dir', 'outputs/colab_results/')
        os.makedirs(results_dir, exist_ok=True)

        result_path = os.path.join(
            results_dir,
            f'{args.command}_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()