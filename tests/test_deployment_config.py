from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_as_non_root_and_exposes_api():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "useradd" in dockerfile
    assert "USER appuser" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "uvicorn" in dockerfile


def test_kubernetes_manifests_have_probes_and_hardening():
    api = (ROOT / "infra/k8s/api.yaml").read_text()
    worker = (ROOT / "infra/k8s/worker.yaml").read_text()
    assert "readOnlyRootFilesystem: true" in api
    assert "allowPrivilegeEscalation: false" in api
    assert "runAsNonRoot: true" in api
    assert "/health/live" in api
    assert "/health/ready" in api
    assert "readOnlyRootFilesystem: true" in worker
    assert "allowPrivilegeEscalation: false" in worker
