package com.example.backgroundstyler.mlkit

import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions

/**
 * ML use #1: ML Kit Selfie Segmentation.
 *
 * ML Kit APIs are regular app-level APIs: images in, [SegmentationMask] out -- no tensors.
 */
class SelfieSegmenterHelper : AutoCloseable {

  // 1. Configure and create the segmenter client (STREAM_MODE = optimized for live camera).
  private val segmenter =
    Segmentation.getClient(
      SelfieSegmenterOptions.Builder()
        .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
        .build()
    )

  /**
   * 2. Prepare the image and run inference. The frame is already upright, so rotation is 0.
   *
   * The returned mask has the same size as [frame]; each value is the foreground confidence in
   * [0, 1].
   */
  fun process(frame: Bitmap): Task<SegmentationMask> =
    segmenter.process(InputImage.fromBitmap(frame, /* rotationDegrees= */ 0))

  override fun close() = segmenter.close()
}
