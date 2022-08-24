
package org.pytorch.LSDnet;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.widget.Toast;

import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;

import java.nio.FloatBuffer;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public abstract class AbstractCameraXActivity<R> extends BaseModuleActivity {
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

    protected static long INPUT_MIN_DELAY = 50;
    protected static long DISPLAY_MIN_DELAY = 40;
    protected static long START_DELAY = 50;
    protected static final int INPUT_QUEUE_SIZE = 9;
    protected static final int DISPLAY_QUEUE_SIZE = 9;

    static class InputImageData {
        protected final FloatBuffer inTensorBuffer;

        InputImageData(FloatBuffer inTensorBuffer) {
            this.inTensorBuffer = inTensorBuffer;
        }
    }

    protected LinkedBlockingQueue<InputImageData> inputImageQueue;
    protected LinkedBlockingQueue<R> outputImageQueue;
    private Bitmap buffer = null;

    protected abstract int getContentViewLayoutId();

    protected abstract TextureView getCameraPreviewTextureView();
    private static final String TAG = "CameraActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getContentViewLayoutId());

        inputImageQueue = new LinkedBlockingQueue<InputImageData>(INPUT_QUEUE_SIZE);
        outputImageQueue = new LinkedBlockingQueue<R>(DISPLAY_QUEUE_SIZE);

        startBackgroundThread();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS,
                    REQUEST_CODE_CAMERA_PERMISSION);
        } else {
            setupCameraX();
        }
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        mProcessingThreadPool.execute(mAnalazeImage);
        mDisplayThreadPool.schedule(mDisplayImage, START_DELAY, TimeUnit.MILLISECONDS);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                        this,
                        "You can't use live video classification example without granting CAMERA permission",
                        Toast.LENGTH_LONG)
                        .show();
                finish();
            } else {
                setupCameraX();
            }
        }
    }

    private void setupCameraX() {
        final TextureView textureView = getCameraPreviewTextureView();
        final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
        final Preview preview = new Preview(previewConfig);
        //preview.setOnPreviewOutputUpdateListener(output -> textureView.setSurfaceTexture(output.getSurfaceTexture()));

        final ImageAnalysisConfig imageAnalysisConfig =
                new ImageAnalysisConfig.Builder()
                        .setTargetResolution(new Size(480, 640))
                        .setCallbackHandler(mBackgroundHandler)
                        .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                        .build();
        final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
        imageAnalysis.setAnalyzer(this::offloadImage);

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }

    private Runnable mAnalazeImage = new Runnable() {
        @Override
        public void run() {
            while (true) {
                analyzeImage();
            }
        }
    };

    private Runnable mDisplayImage = new Runnable() {
        @Override
        public void run() {
            try {
                final R result = outputImageQueue.take();
                runOnUiThread(() -> applyToUiAnalyzeImageResult(result));
                mDisplayThreadPool.schedule(mDisplayImage, DISPLAY_MIN_DELAY, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Log.e("Object Detection", "Error on retrieving output image from queue", e);
            }
        }
    };

    @WorkerThread
    protected abstract void offloadImage(ImageProxy img, int rotationDegrees);

    @WorkerThread
    protected abstract void analyzeImage();

    @UiThread
    protected abstract void applyToUiAnalyzeImageResult(R result);
}