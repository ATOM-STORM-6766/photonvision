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
import java.util.List;
import org.opencv.core.Size;
import org.photonvision.coreml.CoreMLJNI;
import org.photonvision.jni.CoreMLObjectDetector;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.logging.LogGroup;
/** Represents a CoreML *.mlmodel file that can be used for object detection. */
public class CoreMLFileModel implements Model {
    private static final Logger logger = new Logger(CoreMLFileModel.class, LogGroup.Config);


    /** The file containing the model. */
    public final File modelFile;

    /** The labels that the model can detect. */
    public final List<String> labels;

    /** The version of the model. */
    public final CoreMLJNI.ModelVersion version;

    /** The input size of the model. */
    public final Size inputSize;

    /**
     * Creates a new CoreMLModel.
     * Assumes parameters have been validated and parsed by the format handler.
     *
     * @param modelFile The file containing the model.
     * @param labels The labels that the model can detect.
     * @param version The version of the model.
     * @param inputSize The input size required by the model.
     */
    public CoreMLFileModel(File modelFile, List<String> labels, CoreMLJNI.ModelVersion version, Size inputSize) {
        this.modelFile = modelFile;
        this.labels = labels; // Assume labels are already read by handler
        this.version = version;
        this.inputSize = inputSize;

        // Removed logic to parse name, determine version/size, and read labels
        // as this is now handled by CoreMLFileFormatHandler
        logger.info("CoreMLFileModel created for: " + modelFile.getName() + ", Version: " + version + ", Size: " + inputSize);
    }

    @Override
    public String getName() {
        return modelFile.getName();
    }

    @Override
    public ObjectDetector load() {
        // Check if modelFile exists before loading? Or assume handler guarantees it?
        // Assuming handler ensures file exists for now.
        if (!modelFile.exists()) {
            logger.error("Model file does not exist when trying to load: " + modelFile.getPath());
            // Or throw a runtime exception?
            return null; // Or handle error appropriately
        }
        return new CoreMLObjectDetector(this, inputSize);
    }
}
