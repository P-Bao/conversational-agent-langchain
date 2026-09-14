# RAG Retrieval — Monitoring & Observability

Thư mục này version-control toàn bộ cấu hình quan sát cho retrieval service,
không thao tác tay trên UI Grafana.

```
monitoring/
├── dashboards/
│   └── rag-retrieval-dashboard.json   # dashboard export, dùng cho provisioning tự động
├── helm/
│   ├── grafana-values.yaml            # values cho helm chart grafana/grafana (dashboard sidecar)
│   └── servicemonitor.yaml            # CRD Prometheus Operator để tự scrape /metrics
└── README.md                          # file này
```

Kiến trúc chi tiết: xem [`docs/observability.md`](../docs/observability.md).

## ⚠️ Trước khi apply lên cluster (bắt buộc)

Các giá trị sau là **giả định hợp lý** theo pattern `kube-prometheus-stack` phổ
biến — phải xác nhận với đội hạ tầng / Grafana trung tâm trước khi apply:

1. **Chart Grafana** của tổ chức: `grafana/grafana` standalone hay là nằm trong
   `prometheus-community/kube-prometheus-stack`? Sidecar dashboards đã bật chưa?
2. **Label sidecar**: ConfigMap dashboard phải có label khớp với
   `sidecar.dashboards.label` của chart thật (mặc định phổ biến
   `grafana_dashboard=1`).
3. **Label `release` trong servicemonitor.yaml**: phải khớp giá trị mà
   kube-prometheus-stack dùng để tự phát hiện ServiceMonitor (kiểm tra label
   trên Prometheus CR / `prometheus.spec.serviceMonitorSelector` trong cluster).
4. **Namespace** (`monitoring`) và **Service labels** (`app: rag-retrieval`):
   phải khớp deployment thật của service trong k8s.
5. **Named port**: Service của rag-retrieval trong k8s phải expose một port có
   tên `metrics` trỏ tới container port 8005.

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

## Cập nhật dashboard

1. Import `monitoring/dashboards/rag-retrieval-dashboard.json` vào UI Grafana
   (hoặc sửa file JSON trực tiếp).
2. ⚠️ Sau khi build/sửa dashboard, **xác nhận trực tiếp trên UI Grafana**
   (màu sắc, label, giá trị panel) — không chỉ review JSON. Bài học từ
   dashboard chirp3: màu khai báo trong JSON khác màu hiển thị thực tế.
3. Export lại JSON vào `monitoring/dashboards/rag-retrieval-dashboard.json`.
4. `make dashboard-apply` để push lên cluster.

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
