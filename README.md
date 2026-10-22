# Background Styler — "One App, Two Paths"

Demo app for the ADS26_006 session *Building High-Performance On-Device AI Apps
with LiteRT and ML Kit in Google Play services*.

Each camera frame goes through two ML paths:

```
Camera ──frame──► ML use #1: ML Kit Selfie Segmentation ──foreground mask──┐
   │                                                                        ▼
   ├──frame──► ML use #2: LiteRT CompiledModel (GPU, custom .tflite) ──stylized image──► Composed frame
   │                                                                        ▲          (subject kept +
   └──frame─────────────────────────────────────────────────────────────────┘           background styled)
```

| Path | Code | Model |
|---|---|---|
| ML Kit Selfie Segmentation (`STREAM_MODE`) | `mlkit/SelfieSegmenterHelper.kt` | Built into `com.google.mlkit:segmentation-selfie` |
| LiteRT `CompiledModel` + `Accelerator.GPU` | `litert/LiteRtStylizer.kt` | Magenta arbitrary-image-stylization-v1-256 fp16 (`style_predict_fp16.tflite`, `style_transfer_fp16.tflite`) |
| Compositing | `pipeline/FrameComposer.kt` | – |
| Orchestration and timing | `pipeline/StylerViewModel.kt` | – |

ML Kit runs asynchronously on its own executor while LiteRT runs on the ML
thread, so the two paths overlap in time.

## UI

* **Viewfinder** shows the processed frame.
* **Stats overlay** shows per-path latency (ML Kit ms, LiteRT ms and the active
  accelerator), compose time, FPS.
* **Mode switch:** `Composed` (full pipeline) / `ML Kit mask` (path #1 only) /
  `LiteRT style` (path #2 only) / `Original`.
* **Style picker:** four style images. Picking one re-runs the style
  prediction model once.
* **GPU / CPU toggle:** recompiles the CompiledModel live, so you can compare
  latency on stage. If the GPU is unavailable, the app falls back to CPU.
* **Camera flip** switches between front and back cameras.

## LiteRT runtime (branch `gms-core`)

This branch runs the LiteRT CompiledModel API on the **LiteRT runtime in Google Play services**
(`com.google.android.gms:play-services-litert:17.0.0-eap01`, which pulls in `litert-api:2.3.0`).
No LiteRT runtime or GPU accelerator native libraries ship in the APK.

```kotlin
val env = LiteRt.initialize(context, initializationOptions {
  setEnableGpu(true)
  setEnableAutomaticDownload(true)
}).await()
CompiledModel.create(assets, "style_transfer_fp16.tflite", CompiledModel.Options(Accelerator.GPU), env)
```

The EAP SDK is not checked in. Unzip it to `~/gms-litert-eap-maven` (or pass
`-PgmsLitertEapRepo=<dir>`). The EAP AAR is compiled with Kotlin 2.4, so this branch uses
Kotlin 2.4.0. Branch `main` keeps the standalone (bundled `litert:2.2.0`) version.

## Build and run

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Requires JDK 17+ and Android SDK 36. Runs on arm64 devices with API 26 or
higher.

## Assets

* Style-transfer models come from the TensorFlow Lite style transfer example
  (Magenta, Apache 2.0):
  `storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/android/`.
* Style images in `assets/styles/` are AI-generated for this demo.
