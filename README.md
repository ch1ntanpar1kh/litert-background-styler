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
| ML Kit Selfie Segmentation (`STREAM_MODE`, runs on CPU) | `mlkit/SelfieSegmenterHelper.kt` | Built into `com.google.mlkit:segmentation-selfie:16.0.0-beta6` |
| LiteRT `CompiledModel` + `Accelerator.GPU` | `litert/LiteRtStylizer.kt` | Magenta arbitrary-image-stylization-v1-256 fp16 (`style_predict_fp16.tflite`, `style_transfer_fp16.tflite`) |
| Compositing | `pipeline/FrameComposer.kt` | – |
| Orchestration and timing | `pipeline/StylerViewModel.kt` | – |

In `Composed` mode the two paths run at the same time: ML Kit on the CPU (its
default; the bundled selfie segmenter has no GPU option) and the LiteRT style
model on the GPU or CPU, as selected in the UI. Each frame therefore costs about
max(ML Kit, LiteRT) plus compositing, not the sum.

## UI

* **Viewfinder** shows the processed frame.
* **Stats overlay** shows three lines:
  * `#1 ML Kit Selfie Segmentation (CPU)`: ML Kit end-to-end latency, timed on
    its own thread.
  * `#2 LiteRT inference (GPU|CPU)`: `CompiledModel.run()` only, on the active
    accelerator. Shows `CPU` if you pick CPU or the GPU is unavailable.
  * `FPS`: smoothed frame rate of the whole pipeline.

  LiteRT pre/post-processing and compositing times are measured but only logged
  (see [Verify on device](#verify-on-device)). A red error line appears only if
  model loading fails.
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

Compared with the standalone build on `main`, the APK drops `libLiteRt.so`
(5.5 MB) and the GPU accelerator library (3.1 MB), and adds small Play services
client libraries. The APK goes from 58.7 MB to 51.1 MB (about 7.7 MB smaller).

> **Note:** `play-services-litert:17.0.0-eap01` is an early-access SDK and is not
> on Google Maven yet, so this branch only builds if you have the SDK locally.
> Unzip it to `~/gms-litert-eap-maven`, or pass `-PgmsLitertEapRepo=<dir>`. The
> EAP AAR is compiled with Kotlin 2.4, so this branch uses Kotlin 2.4.0.
> The device also needs a Google Play services version that includes the LiteRT
> runtime; otherwise the overlay shows "Play services LiteRT init failed".

Branch `main` keeps the standalone version (bundled `litert:2.2.0`).

## Build and run

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Requires JDK 17 or 21 (Gradle 8.11 does not run on JDK 25+) and Android SDK 36.
Runs on arm64 devices with API 26 or higher.

## Verify on device

```bash
adb shell dumpsys package com.example.backgroundstyler | grep versionName
adb logcat -s PipelineCheck
```

The `PipelineCheck` log shows the app version at startup, which accelerator the
LiteRT style model compiled for (with a warning on CPU fallback), the first ML
Kit mask, and every 60 frames:
`mlkit(CPU)`, `litert-inference(GPU|CPU)`, `litert-pre/post(CPU)` and `compose`
times.

## Assets

* Style-transfer models come from the TensorFlow Lite style transfer example
  (Magenta, Apache 2.0):
  `storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/style_transfer/android/`.
* Style images in `assets/styles/` are AI-generated for this demo.
