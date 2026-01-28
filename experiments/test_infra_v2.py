from src.action.sandbox import DockerSandbox
import os

def test_sandbox_tools():
    print("🧪 Testing V2 Sandbox Infrastructure...")
    
    print(f"Checking socket: {os.path.exists('/var/run/docker.sock')}")
    try:
        sb = DockerSandbox()
        # Hack to force socket if env is missing
        # if not sb.client.api.base_url:
        #    print("Force setting base_url")
        #    sb.client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    except Exception as e:
        import docker
        print(f"Standard init failed: {e}")
        print("Attempting explicit connection to unix://var/run/docker.sock")
        try:
            sb = DockerSandbox()
            sb.client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        except Exception as e2:
             print(f"Explicit connection failed: {e2}")
             return

    with sb:
        print("Container started. Checking tools...")
        
        # Check Ruff
        exit_code, stdout, stderr = sb.exec_run("ruff --version")
        if exit_code == 0:
            print(f"✅ Ruff found: {stdout.strip()}")
        else:
            print(f"❌ Ruff NOT found: {stderr.strip()}")
            
        # Check Mypy
        exit_code, stdout, stderr = sb.exec_run("mypy --version")
        if exit_code == 0:
            print(f"✅ Mypy found: {stdout.strip()}")
        else:
            print(f"❌ Mypy NOT found: {stderr.strip()}")

if __name__ == "__main__":
    test_sandbox_tools()
