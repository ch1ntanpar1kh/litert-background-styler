package com.example.backgroundstyler.litert

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * ML use #2: a custom `.tflite` model run with the LiteRT CompiledModel API on GPU.
 *
 * Uses the open-source Magenta arbitrary-image-stylization-v1-256 (fp16) models:
 * - `style_predict_fp16.tflite`: style image [1,256,256,3] -> style bottleneck [1,1,1,100]. Run once
 *   per style selection.
 * - `style_transfer_fp16.tflite`: content [1,384,384,3] + bottleneck [1,1,1,100] -> stylized image
 *   [1,384,384,3] in [0,1]. Run every camera frame.
 *
 * Not thread-safe: create, use and close from a single thread.
 */
class LiteRtStylizer(private val context: Context) : AutoCloseable {

  private var predictModel: CompiledModel? = null
  private var transferModel: CompiledModel? = null

  // Reused across frames -- no per-frame allocations on the inference path.
  private var transferInputs: List<TensorBuffer> = emptyList()
  private var transferOutputs: List<TensorBuffer> = emptyList()

  private val contentPixels = IntArray(CONTENT_SIZE * CONTENT_SIZE)
  private val contentFloats = FloatArray(CONTENT_SIZE * CONTENT_SIZE * 3)
  private val outputPixels = IntArray(CONTENT_SIZE * CONTENT_SIZE)

  private var styleBottleneck: FloatArray? = null

  /** The accelerator actually in use (may differ from the requested one after fallback). */
  var activeAccelerator: Accelerator? = null
    private set

  val isReady: Boolean
    get() = transferModel != null && styleBottleneck != null

  /**
   * Compiles both models for [preferred], falling back to CPU if the GPU is unavailable.
   *
   * Call off the main thread: accelerator probing and GPU shader compilation take time.
   */
  fun initialize(preferred: Accelerator) {
    closeModels()
    activeAccelerator =
      try {
        loadModels(preferred)
        preferred
      } catch (e: Exception) {
        if (preferred == Accelerator.CPU) throw e
        Log.w(TAG, "Failed to compile for $preferred, falling back to CPU", e)
        closeModels()
        loadModels(Accelerator.CPU)
        Accelerator.CPU
      }
    Log.i(TAG, "Stylizer ready on $activeAccelerator")
  }

  private fun loadModels(accelerator: Accelerator) {
    predictModel =
      CompiledModel.create(
        context.assets,
        PREDICT_MODEL,
        // This line is the entire GPU setup.
        CompiledModel.Options(accelerator),
      )
    val transfer =
      CompiledModel.create(
        context.assets,
        TRANSFER_MODEL,
        CompiledModel.Options(accelerator),
      )
    transferModel = transfer
    transferInputs = transfer.createInputBuffers()
    transferOutputs = transfer.createOutputBuffers()
  }

  /** Computes the style bottleneck for [styleImage]. Call when the user picks a new style. */
  fun setStyle(styleImage: Bitmap) {
    val model = checkNotNull(predictModel) { "initialize() first" }
    val scaled = Bitmap.createScaledBitmap(styleImage, STYLE_SIZE, STYLE_SIZE, true)
    val pixels = IntArray(STYLE_SIZE * STYLE_SIZE)
    val floats = FloatArray(STYLE_SIZE * STYLE_SIZE * 3)
    scaled.getPixels(pixels, 0, STYLE_SIZE, 0, 0, STYLE_SIZE, STYLE_SIZE)
    pixelsToRgbFloats(pixels, floats)

    val inputs = model.createInputBuffers()
    val outputs = model.createOutputBuffers()
    try {
      inputs[0].writeFloat(floats)
      model.run(inputs, outputs)
      styleBottleneck = outputs[0].readFloat()
    } finally {
      inputs.forEach { it.close() }
      outputs.forEach { it.close() }
    }
  }

  /**
   * Stylizes [frame] and returns a [CONTENT_SIZE]x[CONTENT_SIZE] bitmap. The caller scales it to
   * the frame size when compositing.
   */
  fun stylize(frame: Bitmap): Bitmap {
    check(isReady) { "initialize() and setStyle() first" }
    val model = transferModel!!

    // Pre-process: resize to 384x384, RGB float in [0, 1], NHWC.
    val scaled = Bitmap.createScaledBitmap(frame, CONTENT_SIZE, CONTENT_SIZE, true)
    scaled.getPixels(contentPixels, 0, CONTENT_SIZE, 0, 0, CONTENT_SIZE, CONTENT_SIZE)
    pixelsToRgbFloats(contentPixels, contentFloats)

    // Inference (synchronous).
    transferInputs[CONTENT_INPUT].writeFloat(contentFloats)
    transferInputs[BOTTLENECK_INPUT].writeFloat(styleBottleneck!!)
    model.run(transferInputs, transferOutputs)
    val out = transferOutputs[0].readFloat()

    // Post-process: [0, 1] RGB floats -> ARGB pixels.
    for (i in outputPixels.indices) {
      val r = (out[i * 3] * 255f).toInt().coerceIn(0, 255)
      val g = (out[i * 3 + 1] * 255f).toInt().coerceIn(0, 255)
      val b = (out[i * 3 + 2] * 255f).toInt().coerceIn(0, 255)
      outputPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    return Bitmap.createBitmap(outputPixels, CONTENT_SIZE, CONTENT_SIZE, Bitmap.Config.ARGB_8888)
  }

  private fun pixelsToRgbFloats(pixels: IntArray, out: FloatArray) {
    for (i in pixels.indices) {
      val p = pixels[i]
      out[i * 3] = ((p shr 16) and 0xFF) / 255f
      out[i * 3 + 1] = ((p shr 8) and 0xFF) / 255f
      out[i * 3 + 2] = (p and 0xFF) / 255f
    }
  }

  private fun closeModels() {
    transferInputs.forEach { it.close() }
    transferOutputs.forEach { it.close() }
    transferInputs = emptyList()
    transferOutputs = emptyList()
    transferModel?.close()
    predictModel?.close()
    transferModel = null
    predictModel = null
    activeAccelerator = null
  }

  override fun close() {
    closeModels()
    styleBottleneck = null
  }

  companion object {
    private const val TAG = "LiteRtStylizer"
    const val PREDICT_MODEL = "style_predict_fp16.tflite"
    const val TRANSFER_MODEL = "style_transfer_fp16.tflite"
    const val STYLE_SIZE = 256
    const val CONTENT_SIZE = 384
    // Input order of style_transfer_fp16.tflite: [content_image, mobilenet_conv/Conv/BiasAdd].
    private const val CONTENT_INPUT = 0
    private const val BOTTLENECK_INPUT = 1
  }
}
