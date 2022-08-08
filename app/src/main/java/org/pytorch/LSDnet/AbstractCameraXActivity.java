
package org.pytorch.LSDnet;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public abstract class AbstractCameraXActivity<R> extends BaseModuleActivity {
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

    private static long INPUT_MIN_DELAY = 50;
    private static long OUTPUT_MIN_DELAY = 40;
    private static long START_DELAY = 100;
    private static final int INPUT_QUEUE_SIZE = 4;
    private static final int OUTPUT_QUEUE_SIZE = 4;

    private long mLastAnalysisResultTime;

    static class InputImageData {
        protected final Bitmap bitmap;
        private final int rotationDegrees;

        InputImageData(Bitmap bitmap, int rotationDegrees) {
            this.bitmap = bitmap;
            this.rotationDegrees = rotationDegrees;
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
        outputImageQueue = new LinkedBlockingQueue<R>(OUTPUT_QUEUE_SIZE);

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
        imageAnalysis.setAnalyzer((image, rotationDegrees) -> {
            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < INPUT_MIN_DELAY) {
                return;
            }
            if (inputImageQueue.size() < INPUT_QUEUE_SIZE) {
                inputImageQueue.offer(new InputImageData(imgToBitmap(image.getImage()), rotationDegrees));
                mLastAnalysisResultTime = SystemClock.elapsedRealtime();
            }
        });

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }

    protected Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
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
                mDisplayThreadPool.schedule(mDisplayImage, OUTPUT_MIN_DELAY, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Log.e("Object Detection", "Error on retrieving output image from queue", e);
            }
        }
    };

    @WorkerThread
    protected abstract void analyzeImage();

    @UiThread
    protected abstract void applyToUiAnalyzeImageResult(R result);
}