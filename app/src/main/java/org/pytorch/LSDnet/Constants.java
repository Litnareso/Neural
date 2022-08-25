package org.pytorch.LSDnet;

public final class Constants {
    public final static float[] MEAN_RGB = new float[] {0.45f, 0.45f, 0.45f};
    public final static float[] STD_RGB = new float[] {0.225f, 0.225f, 0.225f};
    public final static int COUNT_OF_FRAMES_PER_INFERENCE = 1;
    public final static int CHANNEL_NUM = 3;

    public final static int TARGET_VIDEO_SIZE = 128;
    public final static int MODEL_INPUT_SIZE = COUNT_OF_FRAMES_PER_INFERENCE * 3 * TARGET_VIDEO_SIZE * TARGET_VIDEO_SIZE;
    public final static int TOP_COUNT = 3;

    public final static int RUNNING_MEAN_BUF_SIZE = 10;
}
