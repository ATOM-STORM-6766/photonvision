package org.photonvision.vision.objects;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.coreml.CoreMLJNI;

/** Represents a CoreML *.mlpackage directory that can be used for object detection. */
public class CoreMLPackageModel implements Model {

    private static final Logger logger = new Logger(CoreMLPackageModel.class, LogGroup.Config);

    // Define fields needed for the package model
    public final Path packagePath;
    public final List<String> labels;
    public final CoreMLJNI.ModelVersion version;
    public final Size inputSize;

    /**
     * Creates a new CoreMLPackageModel.
     * Assumes parameters have been validated and parsed by the format handler.
     *
     * @param packagePath The path to the *.mlpackage directory.
     * @param labels The list of labels read from the corresponding labels file.
     * @param version The determined model version.
     * @param inputSize The required input size.
     * @throws IllegalArgumentException (less likely now)
     * @throws IOException (less likely now)
     */
    public CoreMLPackageModel(Path packagePath, List<String> labels, CoreMLJNI.ModelVersion version, Size inputSize)
            throws IllegalArgumentException, IOException {
        this.packagePath = packagePath;
        this.labels = labels;
        this.version = version;
        this.inputSize = inputSize;

        // Ensure packagePath exists and is a directory (basic check)
        if (!Files.isDirectory(packagePath)) {
            throw new IllegalArgumentException("Package path must be an existing directory: " + packagePath);
        }
        // The check for .mlpackage suffix should ideally happen in the handler's supportsPath/loadFromPath

        logger.info("CoreMLPackageModel created for: " + packagePath.getFileName() + ", Version: " + version + ", Size: " + inputSize);

        // Removed logic to parse name, determine version/size, find and read labels file
        // as this is now handled by CoreMLPackageFormatHandler
    }

    @Override
    public String getName() {
        // Return the package directory name
        return packagePath.getFileName().toString();
    }

    @Override
    public ObjectDetector load() {
        // TODO: Implement loading logic for CoreMLObjectDetector using the package path
        // This might require changes in CoreMLObjectDetector or CoreMLJNI to accept a Path
        logger.warn("load() not fully implemented for CoreMLPackageModel - requires JNI/Detector adaptation for packages");

        if (!Files.isDirectory(packagePath)) {
            logger.error("Package path does not exist or is not a directory when trying to load: " + packagePath);
            return null;
        }

        // Placeholder - Needs adaptation for packages
        // Once CoreMLObjectDetector supports loading from a package Path, replace this:
        // Example hypothetical call:
        // return new CoreMLObjectDetector(packagePath, this.version, this.inputSize, this.labels);
        // OR if it still needs the Model object:
        // return new CoreMLObjectDetector(this, inputSize); // Assuming the detector uses model.packagePath internally

        // Returning null for now as the underlying detector likely doesn't support packages yet.
        return null; // Or throw UnsupportedOperationException
    }

} 