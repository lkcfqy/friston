import docker
import os
import time
import tarfile
import io
from typing import Tuple, Optional, Dict

class DockerSandbox:
    """
    Manages a Docker container for safe code execution.
    Acts as the 'body' of the agent where actions have consequences.
    """
    def __init__(self, image: str = "python:3.10-slim", work_dir: str = "/workspace"):
        self.client = docker.from_env()
        self.image = image
        self.work_dir = work_dir
        self.container = None
        
    def start(self):
        """Start a detached container that stays alive."""
        try:
            # Check if image exists, pull if not
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                print(f"Pulling image {self.image}...")
                self.client.images.pull(self.image)
                
            self.container = self.client.containers.run(
                self.image,
                command="tail -f /dev/null", # Keep alive
                detach=True,
                working_dir=self.work_dir,
                # Security hardening (optional but recommended)
                # cap_drop=["ALL"],
                # memory="512m",
            )
            
            # Create workspace dir if not strictly standard (though work_dir usually auto-created)
            self.exec_run(f"mkdir -p {self.work_dir}")
            
            # [V2 Upgrade] Install LSP tools (Ruff, Mypy)
            # In a production agent, we would build a custom Dockerfile.
            # For prototype, we install at runtime (adds ~5-10s startup time).
            print("🛠️ Installing LSP tools (ruff, mypy) in sandbox...")
            exit_code, _, stderr = self.exec_run("pip install ruff mypy", timeout=120)
            if exit_code != 0:
                print(f"⚠️ Failed to install LSP tools: {stderr}")
            else:
                print("✅ LSP tools installed.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start sandbox: {e}")

    def stop(self):
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=1)
                self.container.remove()
            except Exception as e:
                print(f"Warning during cleanup: {e}")
            self.container = None

    def inject_file(self, filename: str, content: str):
        """
        Inject a file into the container.
        Docker API requires tar stream for put_archive.
        """
        if not self.container:
            raise RuntimeError("Container not started")
            
        # Create tar in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            data = content.encode('utf-8')
            tar_info = tarfile.TarInfo(name=filename)
            tar_info.size = len(data)
            tar_info.mtime = time.time()
            tar.addfile(tar_info, io.BytesIO(data))
            
        tar_stream.seek(0)
        
        # Put archive (path is directory to extract to)
        self.container.put_archive(
            path=self.work_dir,
            data=tar_stream
        )

    def exec_run(self, cmd: str, timeout: int = 10) -> Tuple[int, str, str]:
        """
        Execute command and return (exit_code, stdout, stderr).
        Handles timeouts manually since docker-py exec_run timeout is for API call not command duration.
        """
        if not self.container:
            raise RuntimeError("Container not started")
            
        # To handle timeout, we run command in background inside shell and wait
        # But for simplicity in this prototype, we use the blocking exec_run
        # Real-world sandboxes need robust timeout handling (e.g. `timeout 10s cmd`)
        
        wrapped_cmd = f"timeout {timeout}s bash -c '{cmd}'"
        
        try:
            exec_result = self.container.exec_run(
                wrapped_cmd,
                demux=True # Separate stdout/stderr
            )
            exit_code = exec_result.exit_code
            
            stdout = exec_result.output[0] if exec_result.output[0] else b""
            stderr = exec_result.output[1] if exec_result.output[1] else b""
            
            # Check if timeout command itself exited with 124
            if exit_code == 124:
                stderr += b"\nExecution timed out."
                
            return exit_code, stdout.decode('utf-8', errors='ignore'), stderr.decode('utf-8', errors='ignore')
            
        except Exception as e:
            return -1, "", str(e)

    def lint_file(self, filename: str) -> Dict[str, str]:
        """
        Run static analysis (Ruff, Mypy) on a file.
        Returns raw output for parsing.
        """
        if not self.container:
            return {"ruff": "", "mypy": ""}
            
        # Ruff: Output JSON
        rc_r, stdout_r, _ = self.exec_run(f"ruff check {filename} --format=json")
        
        # Mypy: Standard output (easier to parse for now than json flag which changes often)
        # Using --no-error-summary to keep it clean
        rc_m, stdout_m, _ = self.exec_run(f"mypy {filename} --no-error-summary --show-error-codes")

        return {
            "ruff": stdout_r,
            "mypy": stdout_m
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
