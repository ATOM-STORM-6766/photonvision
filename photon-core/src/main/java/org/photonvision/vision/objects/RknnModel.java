/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.objects;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.opencv.core.Size;
import org.photonvision.jni.RknnObjectDetector;
import org.photonvision.rknn.RknnJNI;

public class RknnModel implements Model {
    public final File modelFile;
    public final RknnJNI.ModelVersion version;
    public final List<String> labels;
    public final Size inputSize;

    /**
     * Determines the model version based on the model's filename.
     *
     * <p>"yolov5" -> "YOLO_V5"
     *
     * <p>"yolov8" -> "YOLO_V8"
     *
     * <p>"yolov11" -> "YOLO_V11"
     *
     * @param modelName The model's filename
     * @return The model version
     */
    private static RknnJNI.ModelVersion getModelVersion(String modelName)
            throws IllegalArgumentException {
        if (modelName.contains("yolov5")) {
            return RknnJNI.ModelVersion.YOLO_V5;
        } else if (modelName.contains("yolov8")) {
            return RknnJNI.ModelVersion.YOLO_V8;
        } else if (modelName.contains("yolov11")) {
            return RknnJNI.ModelVersion.YOLO_V11;
        } else {
            throw new IllegalArgumentException("Unknown model version for model " + modelName);
        }
    }

    /**
     * rknn model constructor.
     *
     * @param modelFile path to model on disk. Format: `name-width-height-model.rknn`
     * @param labels path to labels file on disk
     * @throws IllegalArgumentException
     */
    public RknnModel(File modelFile, String labels) throws IllegalArgumentException, IOException {
        this.modelFile = modelFile;

        // parseRKNNName throws an IllegalArgumentException if the model name is invalid
        String[] parts = RknnModel.parseRKNNName(modelFile.getName());

        this.version = getModelVersion(parts[3]);

        int width = Integer.parseInt(parts[1]);
        int height = Integer.parseInt(parts[2]);
        this.inputSize = new Size(width, height);

        try {
            this.labels = Files.readAllLines(Paths.get(labels));
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read labels file " + labels, e);
        }
    }

    public String getName() {
        return modelFile.getName();
    }

    public ObjectDetector load() {
        return new RknnObjectDetector(this, inputSize);
    }

    private static Pattern modelPattern =
            Pattern.compile("^([a-zA-Z0-9._]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.rknn$");

    private static Pattern labelsPattern =
            Pattern.compile("^([a-zA-Z0-9._]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)-labels\\.txt$");

    /**
     * Check naming conventions for models and labels.
     *
     * <p>This is static as it is not dependent on the state of the class.
     *
     * @param modelName the name of the model
     * @param labelsName the name of the labels file
     * @throws IllegalArgumentException if the names are invalid
     */
    public static void verifyRKNNNames(String modelName, String labelsName) {
        // check null
        if (modelName == null || labelsName == null) {
            throw new IllegalArgumentException("Model name and labels name cannot be null");
        }

        // These patterns check that the naming convention of
        // name-widthResolution-heightResolution-modelType is followed

        Matcher modelMatcher = modelPattern.matcher(modelName);
        Matcher labelsMatcher = labelsPattern.matcher(labelsName);

        if (!modelMatcher.matches() || !labelsMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model name and labels name must follow the naming convention of name-widthResolution-heightResolution-modelType.rknn and name-widthResolution-heightResolution-modelType-labels.txt");
        }

        if (!modelMatcher.group(1).equals(labelsMatcher.group(1))
                || !modelMatcher.group(2).equals(labelsMatcher.group(2))
                || !modelMatcher.group(3).equals(labelsMatcher.group(3))
                || !modelMatcher.group(4).equals(labelsMatcher.group(4))) {
            throw new IllegalArgumentException("Model name and labels name must be matching.");
        }
    }

    /**
     * Parse RKNN name and return the name, width, height, and model type.
     *
     * <p>This is static as it is not dependent on the state of the class.
     *
     * @param modelName the name of the model
     * @throws IllegalArgumentException if the model name does not follow the naming convention
     * @return an array containing the name, width, height, and model type
     */
    public static String[] parseRKNNName(String modelName) {
        Matcher modelMatcher = modelPattern.matcher(modelName);

        if (!modelMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model name must follow the naming convention of name-widthResolution-heightResolution-modelType.rknn");
        }

        return new String[] {
            modelMatcher.group(1), modelMatcher.group(2), modelMatcher.group(3), modelMatcher.group(4)
        };
    }
}
