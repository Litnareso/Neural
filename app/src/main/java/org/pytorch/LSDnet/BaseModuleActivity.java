
package org.pytorch.LSDnet;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class BaseModuleActivity extends AppCompatActivity {
    protected ExecutorService mProcessingThreadPool;
    protected ScheduledExecutorService mDisplayThreadPool;
    protected HandlerThread mBackgroundThread;
    protected Handler mBackgroundHandler;
    protected Handler mUIHandler;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mUIHandler = new Handler(getMainLooper());
    }

    @Override
    protected void onPostCreate(@Nullable Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        mProcessingThreadPool = Executors.newSingleThreadExecutor();
        mDisplayThreadPool = Executors.newSingleThreadScheduledExecutor();
        startBackgroundThread();
    }

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("ModuleActivity");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    @Override
    protected void onDestroy() {
        stopBackgroundThread();
        stopProcessingThreadPool();
        stopDisplayingThreadPool();
        super.onDestroy();
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e("Object Detection", "Error on stopping background thread", e);
        }
    }

    protected void stopProcessingThreadPool() {
        mProcessingThreadPool.shutdown();
        try {
            if (!mProcessingThreadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                mProcessingThreadPool.shutdownNow();
                if (!mProcessingThreadPool.awaitTermination(5, TimeUnit.SECONDS))
                    Log.e("Object Detection", "Processing ThreadPool did not terminate");
            }
        } catch (InterruptedException e) {
            Log.e("Object Detection", "Processing ThreadPool did not terminate", e);
            mProcessingThreadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    protected void stopDisplayingThreadPool() {
        mDisplayThreadPool.shutdown();
        try {
            if (!mDisplayThreadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                mDisplayThreadPool.shutdownNow();
                if (!mDisplayThreadPool.awaitTermination(5, TimeUnit.SECONDS))
                    Log.e("Object Detection", "Display ThreadPool did not terminate");
            }
        } catch (InterruptedException e) {
            Log.e("Object Detection", "Display ThreadPool did not terminate", e);
            mDisplayThreadPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}