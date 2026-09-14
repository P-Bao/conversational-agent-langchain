# Observability — Metrics & Tracing cho RAG Retrieval Service

Hai loại quan sát riêng biệt, không thay thế nhau:

- **Metrics (Prometheus)**: số liệu tổng hợp theo thời gian (count, rate,
  latency percentile) hiển thị trên dashboard Grafana.
- **Tracing (OpenTelemetry → Tempo)**: bắt toàn bộ input (prompt gốc, tham số
  truy vấn) và toàn bộ output (đầy đủ danh sách document: `page_content` +
  `metadata` + `score`) của từng request cụ thể, để debug từng case.

## Kiến trúc

```
┌──────────────┐   retrieval (1 span/request)
│ rag-retrieval│──┬──────────────────────────────────► Tempo (trace: query + full docs)
│  (FastAPI)   │  │        OTLP/HTTP (OTel SDK)
│              │  │
│  /metrics    │──┴── Prometheus (scrape /metrics, interval 15s)
└──────────────┘         │
                         ▼
              Grafana trung tâm (dashboard "RAG Retrieval")
```

- Trace được export qua OTel SDK (OTLP/HTTP) tới Tempo / OTel Collector trong
  cluster k8s — endpoint cấu hình qua `OTEL_EXPORTER_OTLP_ENDPOINT`. Không set
  endpoint → tracing là no-op (local dev không cần backend).
- Metrics được scrape tại `GET /metrics` (prometheus_client).
- Code: `src/agent/utils/observability.py` — `TracedRetriever` bọc retriever
  LangChain, nên **cả 3 entry point** (`POST /rag/`, `POST /rag/stream`,
  `POST /semantic/search`) đều được instrument tự động.

## Metrics Prometheus (prefix `rag_retrieval_`)

| Metric | Type | Ý nghĩa |
|---|---|---|
| `rag_retrieval_requests_total` | counter | Tổng số retrieval request. |
| `rag_retrieval_requests_by_day{day="2026-09-14"}` | counter | Số request theo ngày, tính theo **Asia/Ho_Chi_Minh**. Dashboard timezone PHẢI khớp (xem lưu ý 1). |
| `rag_retrieval_requests_by_hour_of_day{hod="14"}` | counter | Số request theo giờ trong ngày, tính theo **Asia/Ho_Chi_Minh**. |
| `rag_retrieval_duration_seconds` | histogram | Full round-trip retrieval (kể cả bước embed query). Buckets: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]` giây. |
| `rag_retrieval_errors_total` | counter | Số request lỗi (exception khi retrieve). |
| `rag_retrieval_requests_in_flight` | gauge | Số request đang xử lý. Tăng lúc bắt đầu, giảm trong `finally` — quay về 0 khi hết tải, không leak khi exception. |
| `rag_retrieval_docs_returned` | histogram | Số lượng doc trả về mỗi request. Cho biết retriever có đang trả rỗng / quá ít / quá nhiều không. Buckets: `[0, 1, 2, 3, 5, 8, 10, 15, 20]`. |

Quy tắc tăng/giảm: `in_flight` tăng lúc bắt đầu, giảm trong `finally`;
`errors_total` tăng khi exception; `duration_seconds` đo full round-trip.

## Tracing — bắt toàn bộ input & output

Mỗi retrieval tạo đúng **1 span** tên `rag.retrieval` với:

- **Input (nguyên văn, không rút gọn/hash)**:
  - `input.query` — prompt/query gốc
  - `input.retriever_name` — tên class retriever
  - `input.top_k` — tham số `k` của retriever
- **Output (toàn bộ danh sách document, không chỉ số lượng)**:
  - `output.num_docs` — số lượng doc (attribute)
  - Span event `retrieval.output` chứa JSON `[{"page_content", "metadata", "score"}, ...]`
    (cắt an toàn ở 8000 ký tự để tránh silent truncation của span attribute)
- **Full payload qua structured log**: loguru ghi dòng
  `rag_retrieval_output` với `trace_id`, `span_id` và **đầy đủ** danh sách
  documents. Join Tempo ↔ Loki bằng `trace_id` trong Grafana
  (trace-to-logs) khi cần xem 100% nội dung.

Service name trên Tempo: `OTEL_SERVICE_NAME` (mặc định `rag-retrieval`).

### Tra cứu 1 request cụ thể

1. Lấy `trace_id` từ log ứng dụng (dòng `rag_retrieval_output`).
2. Mở **Tempo** trong Grafana, tìm theo `trace_id`.
3. Xem span `rag.retrieval`: attributes `input.query`, `input.top_k`; event
   `retrieval.output` chứa danh sách documents.
4. Nếu nội dung bị cắt (docs dài): tìm dòng log `rag_retrieval_output` có
   cùng `trace_id` trong Loki — ở đó có 100% payload.

## Dashboard Grafana — 2 lưu ý bắt buộc (bài học chirp3)

### 1. Timezone cố định, không để `browser`

Dashboard đặt `"timezone": "Asia/Ho_Chi_Minh"` ở cấp dashboard. Backend ghi
label `day`/`hod` theo cùng múi giờ đó. Nếu để `"browser"`, biến
`${__to:date:YYYY-MM-DD}` tính theo giờ trình duyệt người xem, trong khi
backend ghi theo giờ Việt Nam — hai bên lệch nhau, panel kiểu "hôm nay" luôn
trả 0 cho người xem ở timezone khác (lỗi đã gặp ở dashboard chirp3).

Ngoài ra, dashboard tránh dùng biến ngày `${__to:date}` — panel Requests by
Day / by Hour of Day dùng `increase(...[$__range])` group theo label thay vì
filter theo ngày cố định.

### 2. Latency panel giữ giá trị cũ khi idle

`histogram_quantile(rate(..._bucket[5m]))` trả gap cho khoảng thời gian không
có request mới (vì `rate()` không có sample để tính). Xử lý 2 lớp:

**Tầng PromQL (bản chất):** bọc quantile trong `last_over_time` với subquery:

```promql
last_over_time(
  (histogram_quantile(0.95, sum(rate(rag_retrieval_duration_seconds_bucket[5m])) by (le)))[6h:1m]
)
```

Subquery `[6h:1m]` tính lại biểu thức mỗi 1 phút trong 6 giờ gần nhất;
`last_over_time()` lấy điểm gần nhất khác null — panel tiếp tục hiển thị
latency cũ cho tới khi có request mới. Áp dụng cho P50/P95/P99 và cho
docs-returned panel. Nếu hệ thống idle lâu hơn 6h, tăng cửa sổ subquery.

**Tầng panel (hiển thị):** `fieldConfig.defaults.custom` của timeseries đặt
`"spanNulls": true` + `"lineInterpolation": "stepAfter"`. Đây chỉ là hiển thị
— phải xử lý ở tầng PromQL trước, `spanNulls` không tự tạo giá trị.

**KHÔNG** áp dụng `last_over_time` cho request rate: rate về 0 khi hết traffic
là đúng logic; chỉ latency/docs-returned mới cần giữ giá trị cũ ("latency = 0
khi hết traffic" gây hiểu nhầm hệ thống nhanh trong khi thực ra là không có
dữ liệu).

## Deploy dashboard

Xem [`monitoring/README.md`](../monitoring/README.md):
- `make dashboard-configmap` — sinh ConfigMap từ dashboard JSON, gắn label
  `grafana_dashboard=1` cho sidecar tự nhận.
- `make dashboard-apply` — apply ConfigMap + ServiceMonitor lên cluster.

## Cấu hình env

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (trống) | Endpoint OTLP/HTTP của Tempo / OTel Collector trong cluster. Trống = tắt tracing. |
| `OTEL_SERVICE_NAME` | `rag-retrieval` | Service name trên Tempo/Grafana. |
