package com.example.cartoonifyme

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.cartoonifyme.databinding.ActivityMainBinding
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity(), CvCameraViewListener2 {

    private lateinit var binding: ActivityMainBinding
    private var isCartoonOn = false
    private lateinit var lastFrame: Mat

    // Pre-allocate matrices for the cartoonify effect to avoid repeated allocation
    private lateinit var gray: Mat
    private lateinit var edges: Mat
    private lateinit var smoothed: Mat
    private lateinit var bgrFrame: Mat
    private lateinit var result: Mat

    private var processingScale = 0.75 // Process at 75% resolution for better performance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.cameraView.setCvCameraViewListener(this)

        binding.toggleButton.setOnClickListener {
            isCartoonOn = !isCartoonOn
        }

        // Load OpenCV
        if (OpenCVLoader.initLocal()) {
            binding.cameraView.apply {
                setCameraPermissionGranted()
                enableView()
            }
            Log.i("NDKTest", "OpenCV initialized correctly")
        } else {
            Log.e("NDKTest", "Unable to initialize OpenCV")
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        // Initialize all matrices with the camera resolution
        lastFrame = Mat()

        // Initialize matrices for cartoonify function
        val scaledWidth = (width * processingScale).toInt()
        val scaledHeight = (height * processingScale).toInt()

        gray = Mat(scaledHeight, scaledWidth, CvType.CV_8UC1)
        edges = Mat(scaledHeight, scaledWidth, CvType.CV_8UC1)
        bgrFrame = Mat(scaledHeight, scaledWidth, CvType.CV_8UC3)
        smoothed = Mat(scaledHeight, scaledWidth, CvType.CV_8UC3)
        result = Mat(height, width, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        // Release all matrices
        lastFrame.release()
        gray.release()
        edges.release()
        bgrFrame.release()
        smoothed.release()
        result.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        lastFrame = inputFrame.rgba()
        return if (isCartoonOn) cartoonify(lastFrame) else lastFrame
    }

    override fun onResume() {
        super.onResume()
        binding.cameraView.enableView()
    }

    override fun onPause() {
        super.onPause()
        binding.cameraView.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        binding.cameraView.disableView()
    }

    private fun cartoonify(frame: Mat): Mat {
        val scaledInput = Mat()

        // Downscale for processing
        if (processingScale < 1.0) {
            Imgproc.resize(
                frame,
                scaledInput,
                Size(frame.width() * processingScale, frame.height() * processingScale),
                0.0, 0.0, Imgproc.INTER_LINEAR
            )
        } else {
            frame.copyTo(scaledInput)
        }

        // Convert to BGR (3-channel) for bilateral filter
        Imgproc.cvtColor(scaledInput, bgrFrame, Imgproc.COLOR_RGBA2BGR)

        // Convert to grayscale
        Imgproc.cvtColor(scaledInput, gray, Imgproc.COLOR_RGBA2GRAY)

        // Detect edges - use smaller kernel sizes for better performance
        Imgproc.medianBlur(gray, gray, 5) // Reduced from 7 to 5
        Imgproc.adaptiveThreshold(
            gray, edges, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY, 7, 2.0  // Reduced from 9 to 7
        )

        // Smooth colors - use faster parameters for bilateral filter
        // Reduced d from 9 to 7, and sigma values from 75.0 to 50.0
        Imgproc.bilateralFilter(bgrFrame, smoothed, 7, 50.0, 50.0)

        // Convert back to RGBA
        val edgesRgba = Mat()
        Imgproc.cvtColor(edges, edgesRgba, Imgproc.COLOR_GRAY2RGBA)
        Imgproc.cvtColor(smoothed, smoothed, Imgproc.COLOR_BGR2RGBA)

        // Combine edges and smoothed image
        val tempResult = Mat()
        Core.bitwise_and(smoothed, edgesRgba, tempResult)

        // Upscale result to original size if needed
        if (processingScale < 1.0) {
            Imgproc.resize(
                tempResult,
                result,
                frame.size(),
                0.0, 0.0, Imgproc.INTER_LINEAR
            )
        } else {
            tempResult.copyTo(result)
        }

        // Clean up temp matrices
        scaledInput.release()
        edgesRgba.release()
        tempResult.release()

        return result
    }
}