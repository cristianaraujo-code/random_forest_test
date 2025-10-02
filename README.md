# Tutorial to Deploy and Test model.joblib with KServe (local kind environment)

> This version covers everything from scratch: cluster creation (optional), installing components, how to build a custom image with the exact library versions, how to load that image into kind, how to prepare PV/PVC, copy `model.joblib` into the PVC, declare the `InferenceService` using the custom image or the default runtime, how to generate `input_numeric_10.json` (10 pre-transformed instances), and how to expose and test the service using `kubectl port-forward`. It also includes troubleshooting and solutions to the errors you encountered.

Repository: [https://github.com/cristianaraujo-code/random_forest_test](https://github.com/cristianaraujo-code/random_forest_test)

---

## Requirements and Mandatory Versions

**Host tools**: `kubectl`, `kind`, `helm`, `docker`, `curl`, `git`.

**Python (environment to prepare JSON / local tests)** — use exactly these versions:

* `numpy==1.26.4`
* `scikit-learn==1.2.2`
* `pandas==2.3.2`
* `scipy==1.11.4`
* `joblib` (latest compatible)

Create Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
pip install numpy==1.26.4 scikit-learn==1.2.2 pandas==2.3.2 scipy==1.11.4 joblib requests
```

---

## Workflow Summary

1. (Optional) Create a `kind` cluster or use an existing cluster.
2. Install `cert-manager`.
3. Install `Istio` (gateway) — optional ingress controller.
4. Install `KServe` in `RawDeployment` mode (already done, recommended locally).
5. Build a **custom image** based on `kserve/sklearnserver` with exact package versions and load it into `kind`.
6. Create `PV`/`PVC` (hostPath for kind), create `uploader` pod and copy `model.joblib` into the desired folder (e.g. `sklearn-model/model.joblib`).
7. Declare `InferenceService` pointing to `pvc://model-pvc/sklearn-model` and force use of the **custom image** (recommended) or use the default runtime.
8. Generate `input_numeric_10.json` with 10 pre-transformed instances (script included) without modifying `test.csv`.
9. Expose the predictor using `kubectl port-forward service/... 8081:80` and test with `curl`.
10. Debug common errors.

---

## 1) Create kind Cluster (optional)

```bash
kind create cluster --name cluster-kserve
kubectl config use-context kind-cluster-kserve
kubectl get nodes -o wide
```

If pods remain `Pending` due to insufficient memory, recreate the cluster with more resources or increase Docker VM memory.

---

## 2) Install cert-manager

```bash
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.18.2/cert-manager.yaml
kubectl wait --for=condition=available deployment -n cert-manager --all --timeout=120s
```

---

## 3) Install Istio (gateway)

```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update
kubectl create namespace istio-system
helm install istio-base istio/base -n istio-system --create-namespace
helm install istiod istio/istiod -n istio-system
helm install istio-ingress istio/gateway -n istio-system
```

Verify:

```bash
kubectl get pods -n istio-system
kubectl get svc -n istio-system
```

---

## 4) Install KServe (CRDs + controller) in RawDeployment mode

```bash
# Install CRDs
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.15.0

# Create namespace for KServe
kubectl create namespace kserve

# Install KServe with Istio
helm install kserve oci://ghcr.io/kserve/charts/kserve --version v0.15.0 \
  -n kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.controller.gateway.ingressGateway.className=istio
```

Check pods in `kserve` namespace.

---

## 5) Build the Custom Image (CRUCIAL)

**Why**: `model.joblib` requires exact versions of `numpy`, `scikit-learn`, `pandas`, `scipy`. KServe base images may differ. Solution: build a custom image from `kserve/sklearnserver` with pinned versions.

**Dockerfile** (save as `Dockerfile.sklearn-custom`):

```docker
FROM kserve/sklearnserver:v0.15.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.2.2 \
    pandas==2.3.2 \
    scipy==1.11.4

RUN pip install --no-cache-dir joblib
```

**Build and load image into kind**:

```bash
docker build -t sklearn-custom:0.1 -f Dockerfile.sklearn-custom .
kind load docker-image sklearn-custom:0.1 --name cluster-kserve
```

Verify:

```bash
docker images | grep sklearn-custom
```

---

## 6) PV / PVC (hostPath) and Uploader Pod

`pv-local.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-storage
  hostPath:
    path: /tmp/kserve-models
```

`pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: ns-test-model
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: local-storage
```

Apply:

```bash
kubectl create namespace ns-test-model
kubectl apply -f pv-local.yaml
kubectl apply -f pvc.yaml -n ns-test-model
kubectl get pv,pvc -n ns-test-model
```

`uploader-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: uploader
  namespace: ns-test-model
spec:
  containers:
    - name: uploader
      image: busybox
      command: ["/bin/sh", "-c", "sleep 3600"]
      volumeMounts:
        - mountPath: /mnt
          name: modelstore
  volumes:
    - name: modelstore
      persistentVolumeClaim:
        claimName: model-pvc
  restartPolicy: Never
```

Run and copy model:

```bash
kubectl apply -f uploader-pod.yaml
kubectl wait --for=condition=Ready pod/uploader -n ns-test-model --timeout=60s

kubectl exec -n ns-test-model -it pod/uploader -- mkdir -p /mnt/sklearn-model
kubectl cp model.joblib ns-test-model/uploader:/mnt/sklearn-model/model.joblib -n ns-test-model
kubectl exec -n ns-test-model -it pod/uploader -- ls -la /mnt/sklearn-model
```

---

## 7) Declare InferenceService with Custom Image

`inferenceservice-custom-image.yaml`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-model
  namespace: ns-test-model
spec:
  predictor:
    containers:
      - name: kserve-container
        image: sklearn-custom:0.1
        imagePullPolicy: IfNotPresent
        args:
          - --model_dir=/mnt
        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
        volumeMounts:
          - mountPath: /mnt
            name: model-storage
    volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

Apply and check:

```bash
kubectl apply -f inferenceservice-custom-image.yaml
kubectl get inferenceservices -n ns-test-model
kubectl get pods -n ns-test-model
kubectl logs -n ns-test-model deploy/sklearn-model-predictor -f
```

---

## 8) Generate `input_numeric_10.json`

Use the provided script:

```bash
python3 make_input_json_10.py --test test.csv --model model.joblib --out input_numeric_10.json --n 10
```

---

## 9) Expose and Test the Model

Port-forward the predictor service:

```bash
kubectl get svc -n ns-test-model
kubectl port-forward -n ns-test-model service/sklearn-model-predictor 8080:80
```

Test with curl:

```bash
curl -H "Content-Type: application/json" -d @input_numeric_10.json http://127.0.0.1:8080/v1/models/sklearn-model:predict
```

---

## 10) Logs and Debugging

```bash
kubectl logs -n ns-test-model deploy/sklearn-model-predictor -f
```

Common issues:

* Wrong feature count → mismatch in preprocessing.
* String instead of numeric input → ensure JSON is numeric.
* Pending pod → adjust memory requests or increase cluster resources.

---

## 11) Cleanup

```bash
kubectl delete -f inferenceservice-custom-image.yaml -n ns-test-model
kubectl delete pod/uploader -n ns-test-model
kubectl delete pvc/model-pvc -n ns-test-model
kubectl delete pv/model-pv
kubectl delete namespace ns-test-model
```

---

## 12) Final Recommendations

* For production, always bundle the `preprocessor` with the model as a `Pipeline` before exporting with joblib.
* For local testing, the most reliable setup is custom image + `kind load docker-image` + port-forward.