# RAG Retrieval — Monitoring & Observability

Thư mục này version-control toàn bộ cấu hình quan sát cho retrieval service,
không thao tác tay trên UI Grafana.

```
monitoring/
├── dashboards/
│   └── rag-retrieval-dashboard.json   # dashboard export (9 panels: 8 Prometheus + 1 Tempo Recent Traces),
│                                      # dùng cho provisioning tự động
├── helm/
│   ├── grafana-values.yaml            # values cho helm chart grafana/grafana (dashboard sidecar + Tempo datasource)
│   └── servicemonitor.yaml            # CRD Prometheus Operator để tự scrape /metrics (khi service đã trong k8s)
├── k8s/
│   └── rag-retrieval-metrics-scrape.yaml  # Service + Endpoints + ServiceMonitor cho service chạy Docker trên host (pattern chirp3)
└── README.md                          # file này
```

Kiến trúc chi tiết: xem [`docs/observability.md`](../docs/observability.md).

## ⚠️ Trước khi apply lên cluster (bắt buộc)

Các giá trị sau là **giả định hợp lý** theo pattern `kube-prometheus-stack` phổ
biến — phải xác nhận với đội hạ tầng / Grafana trung tâm trước khi apply:

1. **Chart Grafana** của tổ chức: `grafana/grafana` standalone hay là nằm trong
   `prometheus-community/kube-prometheus-stack`? Sidecar dashboards đã bật chưa?
2. **Datasource UID trong dashboard JSON**: dashboard "RAG Retrieval" pin cứng
   Prometheus datasource bằng `"uid": "prometheus"` (mặc định của
   kube-prometheus-stack provisioned datasource — chỉ có 1 Prometheus, không
   sửa được qua UI). ⚠️ Xác nhận UID thật: mở datasource trong UI Grafana, UID
   hiện trong URL (`/connections/datasources/edit/<uid>`). Nếu UID khác
   `prometheus`, sửa lại tất cả `"uid"` trong
   `monitoring/dashboards/rag-retrieval-dashboard.json` cho khớp. Tempo
   datasource đã xác nhận UID `afy8sa3jsx88wf` (dùng khi thêm link/trace panel).
3. **Label sidecar**: ConfigMap dashboard phải có label khớp với
   `sidecar.dashboards.label` của chart thật (mặc định phổ biến
   `grafana_dashboard=1`).
4. **Label `release` trong servicemonitor.yaml**: phải khớp giá trị mà
   kube-prometheus-stack dùng để tự phát hiện ServiceMonitor (kiểm tra label
   trên Prometheus CR / `prometheus.spec.serviceMonitorSelector` trong cluster).
5. **Namespace** (`monitoring`) và **Service labels** (`app: rag-retrieval`):
   phải khớp deployment thật của service trong k8s.
6. **Named port**: Service của rag-retrieval trong k8s phải expose một port có
   tên `metrics` trỏ tới container port 8005.
   datasource đã xác nhận UID `afy8sa3jsx88wf` (dùng khi thêm link/trace panel).
7. **Tempo datasource**: cần có trong Grafana trung tâm để xem trace (đã cấu
   hình thành công qua UI với uid `afy8sa3jsx88wf`, xem chi tiết ở
   `monitoring/helm/grafana-values.yaml`). Hai điểm bắt buộc với Tempo
   self-host qua Helm: (a) URL là in-cluster HTTP API
   `http://tempo.monitoring.svc.cluster.local:3200` — không dùng NodePort;
   (b) **Streaming TẮT cả Search + Metrics queries** — Tempo HTTP API port
   3200 trả về HTTP/1.1, gRPC streaming fail với lỗi
   "http2: frame too large, note that the frame header looked like an
   HTTP/1.1 header".

## Deploy dashboard lên Grafana trung tâm

```bash
# 1. Sinh ConfigMap chứa dashboard JSON, gắn label grafana_dashboard=1
make dashboard-configmap

# 2. Apply ConfigMap + ServiceMonitor
make dashboard-apply
```

Sau vài phút, dashboard "RAG Retrieval" tự xuất hiện trong Grafana (nhờ
dashboard sidecar). Nếu không thấy: kiểm tra namespace/label ở trên có khớp
với chart thật không.

## ⚠️ Service chạy Docker (không phải k8s) — dashboard no data?

ServiceMonitor **chỉ scrape Service trong k8s**. Nếu service rag-retrieval đang
chạy ở Docker trên server (localhost:8005) mà chưa có instance k8s, Prometheus
không có target nào → dashboard no data. Fix theo **pattern chirp3**
(k8s-native, chỉ `kubectl apply` — không đụng release
kube-prometheus-stack): tạo Service không selector + Endpoints trỏ thẳng IP
host + ServiceMonitor scrape Service đó.

```bash
# 1. Lấy IP INTERNAL của server chạy Docker
kubectl get nodes -o wide

# 2. Điền IP vào __NODE_IP__ trong monitoring/k8s/rag-retrieval-metrics-scrape.yaml

# 3. Apply (Service + Endpoints + ServiceMonitor một lần)
make metrics-scrape-apply

# 4. Gửi 1 request tới service (http://<NODE_IP>:8005/rag/), đợi ~15-30s
#    rồi kiểm tra panel dashboard — phải có dữ liệu.
```

Debug nếu vẫn no data: Prometheus UI (`prometheus.monitoring` → Status →
Targets) tìm job của ServiceMonitor `rag-retrieval-metrics` — phải hiện `UP`;
nếu `DOWN` thì firewall chặn hoặc `__NODE_IP__` sai.

Khi service đã được deploy lên k8s (Deployment + Service thật với named port
`metrics`), **xóa `monitoring/k8s/rag-retrieval-metrics-scrape.yaml`** và dùng
`monitoring/helm/servicemonitor.yaml` (scrape Service k8s thật qua selector).

## Cập nhật dashboard

1. Import `monitoring/dashboards/rag-retrieval-dashboard.json` vào UI Grafana
   (hoặc sửa file JSON trực tiếp).
2. ⚠️ Sau khi build/sửa dashboard, **xác nhận trực tiếp trên UI Grafana**
   (màu sắc, label, giá trị panel) — không chỉ review JSON. Bài học từ
   dashboard chirp3: màu khai báo trong JSON khác màu hiển thị thực tế.
3. Export lại JSON vào `monitoring/dashboards/rag-retrieval-dashboard.json`.
4. `make dashboard-apply` để push lên cluster.

⚠️ Panel "Recent Traces" (Tempo) pin cứng datasource UID
`afy8sa3jsx88wf` — nếu Tempo datasource bị re-provision/đổi UID, sửa lại
`"uid"` trong panel id 9 của dashboard JSON cho khớp.

## Hai lưu ý bắt buộc (bài học từ dashboard chirp3)

1. **Timezone cố định `Asia/Ho_Chi_Minh`**: dashboard đặt
   `"timezone": "Asia/Ho_Chi_Minh"` ở cấp dashboard — KHÔNG để `"browser"`.
   Backend ghi label `day`/`hod` theo cùng múi giờ đó. Nếu để `browser`, biến
   `${__to:date:YYYY-MM-DD}` tính theo giờ trình duyệt người xem và lệch với
   label backend (lỗi panel "hôm nay" luôn trả 0 ở dashboard chirp3).
2. **Latency panel giữ giá trị cũ khi idle**: P50/P95/P99 bọc
   `last_over_time((histogram_quantile(...))[6h:1m])` + `spanNulls: true` +
   `lineInterpolation: "stepAfter"`. Không bọc `last_over_time` thì panel bị
   gap khi không có request mới trong cửa sổ `rate()` 5 phút. Nếu hệ thống
   idle lâu hơn 6h, tăng cửa sổ subquery (đổi lại query nặng hơn). KHÔNG áp
   dụng `last_over_time` cho request rate — rate về 0 khi hết traffic là đúng
   logic.
