package org.pytorch.LSDnet;

import static android.graphics.Color.rgb;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.core.app.ActivityCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

import org.pytorch.Device;

public class LiveVideoClassificationActivity extends AbstractCameraXActivity<LiveVideoClassificationActivity.AnalysisResult> {
    private volatile Module mModule = null;
    int model_idx = 0;
    int model_idx_used = 0;
    List<String> model_names = Arrays.asList("model.ptl");
    private ImageView mResultView;
    private TextView mFPSView;
    private int mFrameCount = 0;
    private FloatBuffer inTensorBuffer;



    static class AnalysisResult {
        private final Bitmap bitmap;
        private final long inferenceTime;

        public AnalysisResult(Bitmap bitmap, long inferenceTime) {
            this.bitmap = bitmap;
            this.inferenceTime = inferenceTime;
        }
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
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(this.getApplicationContext(), model_names.get(0)));
            } catch (IOException e) {
                Log.e("Object Detection", "Error on loading model", e);
            }
        }

        final ImageButton buttonswitch = findViewById(R.id.nextbutton);

        buttonswitch.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //final Intent intent = new Intent(MainActivity.this, LiveVideoClassificationActivity.class);
                //startActivity(intent);
                model_idx += 1;
                model_idx = model_idx %  model_names.size();
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

    private Bitmap floatArrayToBitmap(float [] floatArray, int width, int height) {
        // Create empty bitmap in ARGB format
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height * 4];

        // mapping smallest value to 0 and largest value to 255
        float maxValue = -1;
        float minValue = 1;
        for (int i = 0; i < floatArray.length; i++) {
            maxValue = Math.max(floatArray[i], maxValue);
            minValue = Math.min(floatArray[i], minValue);
        }
        float delta = maxValue - minValue;
        // copy each value from float array to RGB channels
        for (int i = 0; i < width* height; i++) {
            int r = (int)((floatArray[i] - minValue) / delta * 255.);
            int g = (int)((floatArray[i+width*height] - minValue) / delta * 255.);
            int b =(int)((floatArray[i+2*width*height] - minValue) / delta * 255.);
            r = Math.min(Math.max(r, 0), 255);
            g = Math.min(Math.max(g, 0), 255);
            b = Math.min(Math.max(b, 0), 255);
            pixels[i] = rgb(r, g, b); // you might need to import for rgb()
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);

        return bmp;
    }

    Bitmap Model_forward(Bitmap buf) {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(buf,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] outimgTensor = mModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        Bitmap tmp = floatArrayToBitmap(outimgTensor, buf.getWidth(),buf.getHeight());
        return tmp;
    }

    @Override
    @WorkerThread
    protected void analyzeImage() {
        if (model_idx != model_idx_used) {
            mModule.destroy();
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(this.getApplicationContext(), model_names.get(model_idx)), null, Device.CPU);
            } catch (IOException e) {

            }
            model_idx_used = model_idx;
        }

//        if (mFrameCount == 0)
//            inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE);

        Bitmap bitmap = null;
        try {
            bitmap = inputImageQueue.take().bitmap;
        } catch (InterruptedException e) {
            Log.e("Object Detection", "Error on retrieving input image from queue", e);
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

//        float ratio = Math.min(bitmap.getWidth(), bitmap.getHeight()) / 160.0f;
//        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, (int)(bitmap.getWidth() / ratio), (int)(bitmap.getHeight() / ratio), true);
//        Bitmap centerCroppedBitmap = Bitmap.createBitmap(resizedBitmap,
//                resizedBitmap.getWidth() > resizedBitmap.getHeight() ? (resizedBitmap.getWidth() - resizedBitmap.getHeight()) / 2 : 0,
//                resizedBitmap.getHeight() > resizedBitmap.getWidth() ? (resizedBitmap.getHeight() - resizedBitmap.getWidth()) / 2 : 0,
//                Constants.TARGET_VIDEO_SIZE, Constants.TARGET_VIDEO_SIZE);
//
//        TensorImageUtils.bitmapToFloatBuffer(centerCroppedBitmap, 0, 0,
//                Constants.TARGET_VIDEO_SIZE, Constants.TARGET_VIDEO_SIZE, Constants.MEAN_RGB, Constants.STD_RGB, inTensorBuffer,
//                (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1) * mFrameCount * Constants.TARGET_VIDEO_SIZE * Constants.TARGET_VIDEO_SIZE);
//
//        mFrameCount++;
//        if (mFrameCount < 4) {
//            return null;
//        }
//
//        mFrameCount = 0;
//        Tensor inputTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, 3, Constants.COUNT_OF_FRAMES_PER_INFERENCE, 160, 160});
//
//        final long startTime = SystemClock.elapsedRealtime();
//        Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
//        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
//
//        final float[] scores = outputTensor.getDataAsFloatArray();
//        Integer scoresIdx[] = new Integer[scores.length];
//        for (int i = 0; i < scores.length; i++)
//            scoresIdx[i] = i;
//
//        Arrays.sort(scoresIdx, new Comparator<Integer>() {
//            @Override public int compare(final Integer o1, final Integer o2) {
//                return Float.compare(scores[o2], scores[o1]);
//            }
//        });
//
//        String tops[] = new String[Constants.TOP_COUNT];
//
//        final String result = String.join(", ", tops);
        bitmap = Bitmap.createScaledBitmap(bitmap, (int)(128), (int)(128), true);
        final long startTime = SystemClock.elapsedRealtime();
        bitmap = Model_forward(bitmap);
        final Bitmap transferredBitmap = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
//        runOnUiThread(new Runnable() {
//            @Override
//            public void run() {
//                to_wait = false;
//                mImageView.setImageBitmap(transferredBitmap);
//                mTextView.setText(str_1);
//            }
//        });

        outputImageQueue.offer(new AnalysisResult(transferredBitmap, inferenceTime));
        return;
    }
}