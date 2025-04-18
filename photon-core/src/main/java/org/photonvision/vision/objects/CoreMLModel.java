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
import org.opencv.core.Size;
import org.photonvision.jni.CoreMLObjectDetector;

/** Represents a CoreML model that can be used for object detection. */
public class CoreMLModel implements Model {
    /** The file containing the model. */
    public final File modelFile;

    /** The labels that the model can detect. */
    public final List<String> labels;

    /** The version of the model. */
    public final ModelVersion version;

    /**
     * Creates a new CoreMLModel.
     *
     * @param modelFile The file containing the model.
     * @param labels The labels that the model can detect.
     * @param version The version of the model.
     */
    public CoreMLModel(File modelFile, String labels) {
        this.modelFile = modelFile;
        this.version = ModelVersion.V1;

        try {
            this.labels = Files.readAllLines(Paths.get(labels));
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read labels file " + labels, e);
        }
    }

    /** The version of the model. */
    public enum ModelVersion {
        /** Version 1 of the model. */
        V1,
        /** Version 2 of the model. */
        V2
    }

    public ObjectDetector load() {
        return new CoreMLObjectDetector(this, new Size(640, 640));
    }

    public String getName() {
        return modelFile.getName();
    }

    public static void verifyCoreMLNames(String modelName, String labelsName) {
        if (!modelName.endsWith(".mlmodel")) {
            throw new IllegalArgumentException("Model name must end with .mlmodel");
        }
    }
}
