package com.example.backgroundstyler.pipeline

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.backgroundstyler.litert.LiteRtStylizer
import com.example.backgroundstyler.mlkit.SelfieSegmenterHelper
import com.google.ai.edge.litert.Accelerator
import com.google.android.gms.tasks.Tasks
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** What the viewfinder shows. [COMPOSED] is the full two-path pipeline. */
enum class ViewMode(val label: String) {
  COMPOSED("Composed"),
  MASK("ML Kit mask"),
  STYLIZED("LiteRT style"),
  ORIGINAL("Original"),
}

data class StyleOption(val name: String, val thumbnail: Bitmap)

data class UiState(
  val frame: Bitmap? = null,
  val mode: ViewMode = ViewMode.COMPOSED,
  val selectedStyle: Int = 0,
  val requestedAccelerator: Accelerator = Accelerator.GPU,
  val activeAccelerator: Accelerator? = null,
  val modelsLoading: Boolean = true,
  val mlKitMs: Long = 0,
  val liteRtMs: Long = 0,
  val composeMs: Long = 0,
  val fps: Float = 0f,
  /** True once ML Kit Selfie Segmentation has returned at least one mask. */
  val mlKitVerified: Boolean = false,
  val error: String? = null,
) {
  /** GPU was requested but LiteRT had to fall back to CPU. */
  val gpuFallback: Boolean
    get() = requestedAccelerator == Accelerator.GPU && activeAccelerator == Accelerator.CPU
}

/**
 * One app, two paths: every camera frame goes to
 * 1. ML Kit Selfie Segmentation -> foreground mask, and
 * 2. a custom LiteRT CompiledModel on GPU -> stylized image,
 * which are then composed into "subject kept + background styled".
 */
class StylerViewModel(private val app: Application) : AndroidViewModel(app) {

  /**
   * All ML work (model compile, style prediction, per-frame inference) runs on this one thread.
   * CameraX delivers frames on it too, so GPU objects are always used from the thread that
   * created them, and frames are naturally dropped while one is being processed.
   */
  val mlExecutor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "ml") }
  private val mlDispatcher = mlExecutor.asCoroutineDispatcher()

  private val segmenter = SelfieSegmenterHelper()
  private val stylizer = LiteRtStylizer(app)
  private val composer = FrameComposer()

  val styles: List<StyleOption> =
    app.assets.list(STYLE_DIR).orEmpty().sorted().map { file ->
      val bmp = app.assets.open("$STYLE_DIR/$file").use { BitmapFactory.decodeStream(it) }
      StyleOption(file.substringBeforeLast('.').replaceFirstChar { it.uppercase() }, bmp)
    }

  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state.asStateFlow()

  @Volatile private var mode = ViewMode.COMPOSED
  private var lastFrameTime = 0L
  private var frameCount = 0L
  private var mlKitLogged = false

  init {
    viewModelScope.launch { loadModels(Accelerator.GPU) }
  }

  fun setMode(newMode: ViewMode) {
    mode = newMode
    _state.update { it.copy(mode = newMode) }
  }

  fun setAccelerator(accelerator: Accelerator) {
    if (accelerator == _state.value.requestedAccelerator) return
    viewModelScope.launch { loadModels(accelerator) }
  }

  fun selectStyle(index: Int) {
    _state.update { it.copy(selectedStyle = index) }
    viewModelScope.launch {
      withContext(mlDispatcher) { runCatching { stylizer.setStyle(styles[index].thumbnail) } }
    }
  }

  private suspend fun loadModels(accelerator: Accelerator) {
    _state.update {
      it.copy(requestedAccelerator = accelerator, modelsLoading = true, error = null)
    }
    try {
      val active =
        withContext(mlDispatcher) {
          stylizer.initialize(accelerator)
          stylizer.setStyle(styles[_state.value.selectedStyle].thumbnail)
          stylizer.activeAccelerator
        }
      _state.update { it.copy(activeAccelerator = active, modelsLoading = false) }
      if (active == accelerator) {
        Log.i(CHECK, "LiteRT CompiledModel style model compiled for $active")
      } else {
        Log.w(CHECK, "LiteRT requested $accelerator but is running on $active (fallback)")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Model load failed", e)
      _state.update { it.copy(modelsLoading = false, error = "Model load failed: ${e.message}") }
    }
  }

  /** CameraX analyzer callback; runs on [mlExecutor]. */
  fun analyze(image: ImageProxy, isFrontCamera: Boolean) {
    val frame =
      try {
        uprightBitmap(image, isFrontCamera)
      } finally {
        image.close()
      }

    val currentMode = mode
    var mlKitMs = 0L
    var liteRtMs = 0L
    var composeMs = 0L
    val output: Bitmap =
      try {
        when (currentMode) {
          ViewMode.ORIGINAL -> frame
          ViewMode.MASK -> {
            val t0 = SystemClock.uptimeMillis()
            val mask = Tasks.await(segmenter.process(frame))
            mlKitMs = SystemClock.uptimeMillis() - t0
            onMaskReceived(mask.width, mask.height)
            composer.maskOverlay(frame, mask)
          }
          ViewMode.STYLIZED -> {
            if (!stylizer.isReady) frame
            else {
              val t0 = SystemClock.uptimeMillis()
              val styled = stylizer.stylize(frame)
              liteRtMs = SystemClock.uptimeMillis() - t0
              Bitmap.createScaledBitmap(styled, frame.width, frame.height, true)
            }
          }
          ViewMode.COMPOSED -> {
            if (!stylizer.isReady) frame
            else {
              // Both paths run concurrently: ML Kit on its own executor, LiteRT on this thread.
              val t0 = SystemClock.uptimeMillis()
              var maskDoneAt = 0L
              val maskTask =
                segmenter.process(frame).addOnCompleteListener(Runnable::run) {
                  maskDoneAt = SystemClock.uptimeMillis()
                }
              val styled = stylizer.stylize(frame)
              liteRtMs = SystemClock.uptimeMillis() - t0
              val mask = Tasks.await(maskTask)
              // ML Kit's own latency (it ran in parallel with LiteRT).
              mlKitMs = (if (maskDoneAt > 0) maskDoneAt else SystemClock.uptimeMillis()) - t0
              onMaskReceived(mask.width, mask.height)
              val t1 = SystemClock.uptimeMillis()
              val composed = composer.compose(frame, styled, mask)
              composeMs = SystemClock.uptimeMillis() - t1
              composed
            }
          }
        }
      } catch (e: Exception) {
        Log.e(TAG, "Frame processing failed", e)
        frame
      }

    if (++frameCount % 60 == 0L && currentMode == ViewMode.COMPOSED) {
      Log.i(
        CHECK,
        "frame=$frameCount mlkit=${mlKitMs}ms litert(${stylizer.activeAccelerator})=${liteRtMs}ms " +
          "compose=${composeMs}ms",
      )
    }

    val now = SystemClock.uptimeMillis()
    val instFps = if (lastFrameTime > 0) 1000f / (now - lastFrameTime).coerceAtLeast(1) else 0f
    lastFrameTime = now
    _state.update {
      it.copy(
        frame = output,
        mlKitMs = mlKitMs,
        liteRtMs = liteRtMs,
        composeMs = composeMs,
        fps = if (it.fps == 0f) instFps else it.fps * 0.9f + instFps * 0.1f,
      )
    }
  }

  private fun onMaskReceived(width: Int, height: Int) {
    if (mlKitLogged) return
    mlKitLogged = true
    Log.i(CHECK, "ML Kit Selfie Segmentation (STREAM_MODE) returned mask ${width}x$height")
    _state.update { it.copy(mlKitVerified = true) }
  }

  private fun uprightBitmap(image: ImageProxy, mirror: Boolean): Bitmap {
    val src = image.toBitmap()
    val rotation = image.imageInfo.rotationDegrees
    if (rotation == 0 && !mirror) return src
    val matrix =
      Matrix().apply {
        postRotate(rotation.toFloat())
        if (mirror) postScale(-1f, 1f)
      }
    return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
  }

  override fun onCleared() {
    mlExecutor.execute {
      stylizer.close()
      segmenter.close()
    }
    mlExecutor.shutdown()
  }

  private companion object {
    const val TAG = "StylerViewModel"
    /** `adb logcat -s PipelineCheck` shows which ML paths are live and on which accelerator. */
    const val CHECK = "PipelineCheck"
    const val STYLE_DIR = "styles"
  }
}
