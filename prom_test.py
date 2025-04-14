import subprocess

def run_oc_command(cmd_args):
    result = subprocess.run(["oc"] + cmd_args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout.strip()

# 1. Get the pod name for prometheus-k8s (adjust namespace if different)
def get_prometheus_pod():
    pod = run_oc_command(["get", "pods", "-n", "openshift-monitoring", "-l", "app=prometheus", "-o", "jsonpath={.items[0].metadata.name}"])
    return pod

def get_prometheus_k8s_pod():
    pods = run_oc_command([
        "get", "pods",
        "-n", "openshift-monitoring",
        "-o", "jsonpath={.items[*].metadata.name}"
    ])
    return pods
    # Filter for prometheus-k8s pods
    for pod in pods.split():
        if pod.startswith("prometheus-k8s"):
            return pod
    return None

# 2. Run a curl command inside the prometheus pod
def run_curl_in_pod(pod_name, url):
    return run_oc_command([
        "exec", "-n", "openshift-monitoring", pod_name, "--",
        "curl", "-s", url
    ])

if __name__ == "__main__":
    pod = get_prometheus_pod()
    print(f"Found Prometheus pod: {pod}")

    url = "http://localhost:9090/metrics"  # or whatever endpoint you're targeting
    output = run_curl_in_pod(pod, url)
    print("Curl output:\n", output)

