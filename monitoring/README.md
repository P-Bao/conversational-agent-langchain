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
