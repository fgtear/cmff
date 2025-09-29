import os
import mlflow
import shutil
import logging
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def log_files(logger, code_dir, save_dir="", target_extensions=None, ignore_dirs=None):
    """将指定目录下的代码文件记录到MLflow中，使用传入的mlflow logger实例。
    Args:
    logger: MLflow logger实例
    code_dir: 代码根目录
    save_dir: 默认保存在artifact_path下的目录。如果save_dir为"code"，则实际保存路径为"artifact/code"
    target_extensions: 要记录的文件扩展名列表，默认为['.py']
    ignore_dirs: 要忽略的目录列表，默认为['.git', '__pycache__', 'mlruns', '.ipynb_checkpoints']
    """
    if target_extensions is None:
        target_extensions = [".py"]

    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            ".vscode",
            ".cache",
            ".venv",
            "__pycache__",
            "mlruns",
            ".ipynb_checkpoints",
        ]

    for foldername, subfolders, filenames in os.walk(code_dir):
        # 检查当前目录是否应该被忽略
        if any(ignore_dir in foldername.split(os.sep) for ignore_dir in ignore_dirs):
            continue

        for filename in filenames:
            if any(filename.endswith(ext) for ext in target_extensions):
                # 构建文件的完整路径
                file_path = os.path.join(foldername, filename)
                # 计算相对于代码目录的相对路径
                rel_path = os.path.relpath(file_path, code_dir)
                # 获取相对路径的目录部分
                artifact_subdir = os.path.dirname(rel_path)
                # 构建正确的artifact路径，避免末尾斜杠问题
                artifact_path = save_dir
                if artifact_subdir:
                    artifact_path = os.path.normpath(os.path.join(save_dir, artifact_subdir))
                # 将文件记录为Artifact，保留目录结构
                logger.experiment.log_artifact(
                    run_id=logger.run_id,
                    local_path=file_path,
                    artifact_path=artifact_path,
                )


def delete_artifacts_directory(artifact_dir_name, mlruns_path="mlruns"):
    """删除mlruns下所有run的artifacts中指定的目录及其文件。

    Args:
        artifact_dir_name: 要删除的artifacts目录名称
        mlruns_path: mlruns的路径，默认为"mlruns"

    Returns:
        删除的目录数量
    """
    if not os.path.exists(mlruns_path):
        print(f"MLflow runs目录 '{mlruns_path}' 不存在")
        return 0

    deleted_count = 0

    # 遍历mlruns下的所有实验ID目录
    for exp_id in os.listdir(mlruns_path):
        exp_path = os.path.join(mlruns_path, exp_id)

        # 跳过非目录文件和特殊的元数据文件
        if not os.path.isdir(exp_path) or exp_id == ".trash" or exp_id.startswith("."):
            continue

        # 遍历实验下的所有run
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)

            if not os.path.isdir(run_path):
                continue

            # 检查run目录下是否有artifacts目录
            artifacts_path = os.path.join(run_path, "artifacts")
            if os.path.exists(artifacts_path):
                # 检查是否有待删除的目录
                target_dir = os.path.join(artifacts_path, artifact_dir_name)
                if os.path.exists(target_dir):
                    try:
                        shutil.rmtree(target_dir)
                        deleted_count += 1
                        print(f"已删除: {target_dir}")
                    except Exception as e:
                        print(f"删除 {target_dir} 时出错: {str(e)}")

    print(f"总共删除了 {deleted_count} 个 '{artifact_dir_name}' 目录")
    return deleted_count


def clean_deleted_artifacts(artifact_root: str, tracking_uri: str = None, dry_run: bool = True):
    """
    清理文件系统上那些对应于已在 MLflow UI/后端删除的实验的 artifact 目录。
    只删除run目录下的artifacts文件夹，保留run目录本身和其他元数据。

    重要提示：此操作会永久删除文件！请务必先在 `dry_run=True` 模式下运行，
    并确保备份了重要数据。

    Args:
        artifact_root (str): MLflow 存储 artifact 的根目录路径。
                             对于本地文件存储，这通常是 'mlruns' 目录的路径，
                             或者是启动 MLflow Server 时用 --default-artifact-root 指定的路径。
                             例如: "/home/user/mlruns" 或 "C:/Users/user/mlruns"。
        tracking_uri (str, optional): MLflow tracking server 的 URI。
                                      如果为 None，MLflow 会使用默认配置（通常是环境变量
                                      MLFLOW_TRACKING_URI 或本地的 ./mlruns）。
                                      例如: "http://localhost:5000" 或 "file:///path/to/mlruns"。
                                      Defaults to None.
        dry_run (bool): 如果为 True，则仅打印将要删除的目录，而不会实际执行删除操作。
                        强烈建议先使用 True 运行。 Defaults to True.
    """
    if not os.path.isdir(artifact_root):
        logging.error(f"指定的 Artifact 根目录不存在或不是一个目录: {artifact_root}")
        return

    logging.info(f"--- 开始清理 MLflow Artifacts ---")
    logging.info(f"Artifact 根目录: {artifact_root}")
    logging.info(f"Tracking URI: {tracking_uri if tracking_uri else '默认'}")
    logging.info(f"Dry Run 模式: {'启用' if dry_run else '禁用 - 将实际删除文件!'}")

    try:
        # 1. 初始化 MLflow Client
        client = MlflowClient(tracking_uri=tracking_uri)
        effective_tracking_uri = client.tracking_uri
        logging.info(f"成功连接到 MLflow Tracking Server: {effective_tracking_uri}")

        # 检查 Tracking URI 是否为本地文件路径，这通常意味着 artifact_root 应该与之匹配或在其之下
        # 这只是一个辅助检查，主要依赖用户提供的 artifact_root
        if effective_tracking_uri.startswith("file://"):
            expected_mlruns_path = effective_tracking_uri[len("file://") :]
            # 确保 artifact_root 是绝对路径以便比较
            abs_artifact_root = os.path.abspath(artifact_root)
            if not abs_artifact_root.startswith(os.path.abspath(expected_mlruns_path)):
                logging.warning(
                    f"提供的 artifact_root '{abs_artifact_root}' "
                    f"似乎不在 tracking URI 指向的目录 '{expected_mlruns_path}' 之下。请确认路径是否正确。"
                )

    except Exception as e:
        logging.error(f"无法连接到 MLflow Tracking Server 或初始化 Client 时出错: {e}")
        return

    # 2. 获取所有活跃的实验和运行 ID
    active_experiments = {}  # 字典：{experiment_id: set(run_id)}
    try:
        # 只获取活跃的实验 (view_type=ViewType.ACTIVE_ONLY)
        experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        active_experiment_ids = {exp.experiment_id for exp in experiments}
        logging.info(f"从 MLflow 中找到 {len(active_experiment_ids)} 个活跃的实验。")

        # 对于每个活跃的实验，获取其下所有活跃的运行
        for exp_id in active_experiment_ids:
            # 只获取活跃的运行 (run_view_type=ViewType.ACTIVE_ONLY)
            active_runs = client.search_runs(experiment_ids=[exp_id], run_view_type=ViewType.ACTIVE_ONLY)
            active_experiments[exp_id] = {run.info.run_uuid for run in active_runs}
        logging.info(f"已获取 {len(active_experiments)} 个活跃实验下的活跃运行信息。")

    except Exception as e:
        logging.error(f"查询 MLflow 实验或运行时出错: {e}")
        return

    # 3. 扫描文件系统上的 Artifact 目录
    artifacts_to_delete = []
    logging.info(f"开始扫描文件系统 Artifact 目录: {artifact_root}")

    try:
        # 列出 artifact_root 下的所有项目（主要是实验 ID 目录）
        fs_experiment_dirs = [d for d in os.listdir(artifact_root) if os.path.isdir(os.path.join(artifact_root, d))]
    except OSError as e:
        logging.error(f"读取 Artifact 根目录时出错 {artifact_root}: {e}")
        return

    for exp_dir_name in fs_experiment_dirs:
        exp_dir_path = os.path.join(artifact_root, exp_dir_name)

        # 跳过 .trash 目录
        if exp_dir_name == ".trash":
            logging.info(f"检测到 '.trash' 目录，跳过: {exp_dir_path}")
            continue

        # 检查这个实验 ID 是否在活跃实验列表中
        if exp_dir_name not in active_experiment_ids:
            # 如果实验不活跃，我们只查找并清理其下所有run目录中的artifacts文件夹
            logging.info(f"发现非活跃实验目录 '{exp_dir_name}'，将检查其下所有run的artifacts目录。")

            try:
                # 列出实验目录下的所有项目（主要是运行 ID 目录）
                fs_run_dirs = [d for d in os.listdir(exp_dir_path) if os.path.isdir(os.path.join(exp_dir_path, d))]
            except OSError as e:
                logging.warning(f"读取实验目录时出错 {exp_dir_path}: {e}。将跳过检查此实验下的运行。")
                continue

            for run_dir_name in fs_run_dirs:
                run_dir_path = os.path.join(exp_dir_path, run_dir_name)

                # 构建可能的artifacts目录路径
                artifacts_dir_path = os.path.join(run_dir_path, "artifacts")
                if os.path.isdir(artifacts_dir_path):
                    logging.info(f"[待清理] 在非活跃实验 '{exp_dir_name}' 的运行 '{run_dir_name}' 中找到artifacts目录")
                    artifacts_to_delete.append(artifacts_dir_path)

            continue  # 处理完非活跃实验，继续下一个实验

        # 如果实验是活跃的，检查其下的运行目录
        active_run_ids_for_exp = active_experiments.get(exp_dir_name, set())
        try:
            # 列出实验目录下的所有项目（主要是运行 ID 目录）
            fs_run_dirs = [d for d in os.listdir(exp_dir_path) if os.path.isdir(os.path.join(exp_dir_path, d))]
        except OSError as e:
            logging.warning(f"读取实验目录时出错 {exp_dir_path}: {e}。将跳过检查此实验下的运行。")
            continue

        for run_dir_name in fs_run_dirs:
            run_dir_path = os.path.join(exp_dir_path, run_dir_name)

            # 运行 ID 通常是 32 位的十六进制字符串
            is_potential_run_id = len(run_dir_name) == 32 and all(c in "0123456789abcdefABCDEF" for c in run_dir_name)

            if not is_potential_run_id:
                logging.debug(f"在实验 '{exp_dir_name}' 下发现非标准运行 ID 格式的目录 '{run_dir_name}'，跳过。")
                continue

            # 检查这个运行 ID 是否在该实验的活跃运行列表中
            if run_dir_name not in active_run_ids_for_exp:
                # 只删除非活跃运行的artifacts目录
                artifacts_dir_path = os.path.join(run_dir_path, "artifacts")
                if os.path.isdir(artifacts_dir_path):
                    logging.info(
                        f"[待清理] artifacts目录 (运行 '{run_dir_name}', 实验 '{exp_dir_name}') 存在于文件系统，但在 MLflow 中不是活跃运行。"
                    )
                    artifacts_to_delete.append(artifacts_dir_path)

    # 4. 执行清理操作 (或预览)
    if not artifacts_to_delete:
        logging.info("扫描完成，没有找到需要清理的已删除 artifact 目录。")
        return

    logging.info(f"\n--- {'预览模式 - 不会删除文件' if dry_run else '准备执行删除操作'} ---")
    logging.info(f"共找到 {len(artifacts_to_delete)} 个可以清理的artifacts目录。")
    for artifact_path in artifacts_to_delete:
        logging.info(f"  - {'[预览]' if dry_run else '[将删除]'} {artifact_path}")

    if not dry_run:
        # 再次确认！
        confirm = input(
            "\n警告：这将永久删除上面列出的所有artifacts目录及其内容！\n请仔细检查列表。是否确定要继续？ (输入 'yes' 以确认): "
        )
        if confirm.lower() == "yes":
            deleted_count = 0
            error_count = 0
            logging.info("开始执行删除...")
            for artifact_path in artifacts_to_delete:
                try:
                    shutil.rmtree(artifact_path)
                    logging.info(f"  已删除: {artifact_path}")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"  删除失败: {artifact_path} - {e}")
                    error_count += 1
            logging.info(f"删除操作完成。成功删除 {deleted_count} 个artifacts目录，失败 {error_count} 个。")
        else:
            logging.info("删除操作已取消。")
    else:
        logging.info("\nDry run 完成。没有文件被删除。如果列表正确，请使用 `dry_run=False` 再次运行以执行删除。")

    logging.info("--- 清理 MLflow Artifacts 结束 ---")


if __name__ == "__main__":
    # os.system("mlflow gc") # 清理mlruns下的已删除实验和运行
    # delete_artifacts_directory("checkpoints")

    clean_deleted_artifacts(artifact_root="mlruns", tracking_uri=None, dry_run=False)
