package org.pytorch.LSDnet;

import static android.graphics.Color.rgb;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.pytorch.Device;

public class LiveVideoClassificationActivity extends AbstractCameraXActivity<LiveVideoClassificationActivity.AnalysisResult> {
    private volatile Module mModule = null;
    volatile int model_idx = 0;
    volatile int model_idx_used = -1;
    List<ProcessingModel> models = Arrays.asList(
            new ProcessingModel("model.ptl", 1, 128, 128, 50, 40),
            new ProcessingModel("model_1.ptl", 1, 128, 128, 40, 30),
            new ProcessingModel("model_3.ptl", 3, 128, 128, 25, 25)
    );
    private ImageView mResultView;
    private TextView mFPSView;
    private final AtomicInteger mFrameCount = new AtomicInteger();
    private FloatBuffer inTensorBuffer;

    private long mLastAnalysisResultTime = 0;

    static class ProcessingModel {
        private final String fileName;
        private final int framesPerInterference;
        private final int resolutionWidth;
        private final int resolutionHeight;
        private final long inputDelay;
        private final long displayDelay;

        ProcessingModel(String fileName, int framesPerInterference, int resolutionWidth, int resolutionHeight, long inputDelay, long displayDelay) {
            this.fileName = fileName;
            this.framesPerInterference = framesPerInterference;
            this.resolutionWidth = resolutionWidth;
            this.resolutionHeight = resolutionHeight;
            this.inputDelay = inputDelay;
            this.displayDelay = displayDelay;
        }

        public int getFrameSize() {
            return resolutionHeight * resolutionWidth * Constants.CHANNEL_NUM;
        }

        public int getInputSize() {
            return getFrameSize() * framesPerInterference;
        }
    }

    static class AnalysisResult {
        private final Bitmap bitmap;
        private final long inferenceTime;

        public AnalysisResult(Bitmap bitmap, long inferenceTime) {
            this.bitmap = bitmap;
            this.inferenceTime = inferenceTime;
        }
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

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_live_video_classification;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (mModule == null) {
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(this.getApplicationContext(), models.get(0).fileName));
            } catch (IOException e) {
                Log.e("Object Detection", "Error on loading model", e);
            }
        }

        final ImageButton buttonswitch = findViewById(R.id.nextbutton);

        buttonswitch.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //final Intent intent = new Intent(MainActivity.this, LiveVideoClassificationActivity.class);
                //startActivity(intent);
                model_idx = (model_idx + 1) % models.size();
            }
        });
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        mFPSView = findViewById(R.id.fpsView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        final String str_1 = String.format("Time: %dms", result.inferenceTime);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mResultView.setImageBitmap(result.bitmap);
                mFPSView.setText(str_1);
            }
        });
//        mResultView.setText(result.);
//        // TODO
//        mResultView.invalidate();
    }

    private Bitmap floatArrayToBitmap(float[] floatArray, int width, int height) {
        // Create empty bitmap in ARGB format
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        // mapping smallest value to 0 and largest value to 255
        float maxValue = -1;
        float minValue = 1;
        for (int i = 0; i < floatArray.length; i++) {
            maxValue = Math.max(floatArray[i], maxValue);
            minValue = Math.min(floatArray[i], minValue);
        }
        float delta = maxValue - minValue;

        for (int idx = 0; idx < floatArray.length; idx++) {
            floatArray[idx] = (float) ((floatArray[idx] - minValue) / delta * 255.);
        }
        for (int i = 0; i < width * height; i++) {
            int r = (int) (floatArray[i]);
            int g = (int) (floatArray[i + width * height]);
            int b = (int) (floatArray[i + 2 * width * height]);
            pixels[i] = rgb(r, g, b);
        }

        // TODO benchmark
        // copy each value from float array to RGB channels
//        for (int i = 0; i < width * height; i++) {
//            int r = (int) ((floatArray[i] - minValue) / delta * 255.);
//            int g = (int) ((floatArray[i + width * height] - minValue) / delta * 255.);
//            int b = (int) ((floatArray[i + 2 * width * height] - minValue) / delta * 255.);
//            r = Math.min(Math.max(r, 0), 255);
//            g = Math.min(Math.max(g, 0), 255);
//            b = Math.min(Math.max(b, 0), 255);
//            pixels[i] = rgb(r, g, b); // you might need to import for rgb()
//        }
        // TODO process all the frames in the batch in one go
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);

        return bmp;
    }

    Bitmap Model_forward(Bitmap buf) {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(buf,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] outimgTensor = mModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        return floatArrayToBitmap(outimgTensor, buf.getWidth(), buf.getHeight());
    }

    @Override
    @WorkerThread
    protected void offloadImage(ImageProxy image, int rotationDegrees) {
        if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < INPUT_MIN_DELAY) {
            return;
        }
        if (inputImageQueue.size() < INPUT_QUEUE_SIZE) {
            mLastAnalysisResultTime = SystemClock.elapsedRealtime();
            Bitmap bitmap = imgToBitmap(image.getImage());
            Matrix matrix = new Matrix();
            matrix.postRotate(90.0f);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap = Bitmap.createScaledBitmap(bitmap, models.get(model_idx_used).resolutionWidth, models.get(model_idx_used).resolutionHeight, true);

            TensorImageUtils.bitmapToFloatBuffer(bitmap, 0, 0,
                    models.get(model_idx_used).resolutionWidth, models.get(model_idx_used).resolutionHeight,
                    Constants.MEAN_RGB, Constants.STD_RGB, inTensorBuffer,
                    (models.get(model_idx_used).framesPerInterference - 1) * mFrameCount.get() *
                            models.get(model_idx_used).resolutionHeight *
                            models.get(model_idx_used).resolutionWidth);

            mFrameCount.getAndIncrement();
            if (mFrameCount.get() < models.get(model_idx_used).framesPerInterference) {
                return;
            }
            mFrameCount.set(0);

            inputImageQueue.offer(new InputImageData(inTensorBuffer));
        }

    }

    @Override
    @WorkerThread
    protected void analyzeImage() {
        if (model_idx != model_idx_used) {
            model_idx_used = model_idx;
            mModule.destroy();
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(this.getApplicationContext(), models.get(model_idx_used).fileName), null, Device.CPU);
            } catch (IOException e) {
                Log.e("Object Detection", "Error on loading new model", e);
            }
            inputImageQueue.clear();
            mFrameCount.set(0);
            inTensorBuffer = Tensor.allocateFloatBuffer(models.get(model_idx_used).getInputSize());
            DISPLAY_MIN_DELAY = models.get(model_idx_used).displayDelay;
            INPUT_MIN_DELAY = models.get(model_idx_used).inputDelay;
        }
        FloatBuffer tensorBuffer = null;
        try {
            tensorBuffer = inputImageQueue.take().inTensorBuffer;
        } catch (InterruptedException e) {
            Log.e("Object Detection", "Error on retrieving input image from queue", e);
        }
        final long startTime = SystemClock.elapsedRealtime();

        Tensor inputTensor = Tensor.fromBlob(tensorBuffer, new long[]{
                1,
                (long) Constants.CHANNEL_NUM * models.get(model_idx_used).framesPerInterference,
                models.get(model_idx_used).resolutionWidth,
                models.get(model_idx_used).resolutionHeight
        });

        final float[] outimgTensor = mModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        for (int counter = 0; counter < models.get(model_idx_used).framesPerInterference; counter++) {
            final float[] frame = Arrays.copyOfRange(outimgTensor,
                    models.get(model_idx_used).getFrameSize() * counter,
                    models.get(model_idx_used).getFrameSize() * (counter + 1));
            final Bitmap tmp = floatArrayToBitmap(frame, models.get(model_idx_used).resolutionWidth, models.get(model_idx_used).resolutionHeight);
            final Bitmap transferredBitmap = Bitmap.createScaledBitmap(tmp, 512, 512, true);
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            outputImageQueue.offer(new AnalysisResult(transferredBitmap, inferenceTime));
        }

        return;
    }
}