import pytest
import docker
from src.action.sandbox import DockerSandbox

def test_sandbox_lifecycle():
    """Test start and stop."""
    try:
        docker.from_env().ping()
    except Exception:
        pytest.skip("Docker not available")

    sb = DockerSandbox()
    sb.start()
    assert sb.container is not None
    assert sb.container.status == 'created' or sb.container.status == 'running'
    
    sb.stop()
    # Verify removal involves checking client or ensuring no error
    # assert that accessing sb.container.reload() raises error or container is gone

def test_file_injection_and_exec():
    try:
        docker.from_env().ping()
    except Exception:
        pytest.skip("Docker not available")
        
    code = "print('Hello from Sandbox')"
    filename = "test_script.py"
    
    with DockerSandbox() as sb:
        sb.inject_file(filename, code)
        
        # Verify file exists
        code, out, err = sb.exec_run(f"ls {filename}")
        assert code == 0
        
        # Run python script
        code, out, err = sb.exec_run(f"python {filename}")
        assert code == 0
        assert "Hello from Sandbox" in out

def test_timeout_mechanism():
    try:
        docker.from_env().ping()
    except Exception:
        pytest.skip("Docker not available")
        
    with DockerSandbox() as sb:
        # Sleep for 2 seconds, timeout 1 second
        code, out, err = sb.exec_run("sleep 2", timeout=1)
        assert code == 124 # Timeout exit code
