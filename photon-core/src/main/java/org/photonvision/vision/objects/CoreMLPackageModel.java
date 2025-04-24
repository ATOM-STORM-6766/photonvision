package org.photonvision.vision.objects;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.coreml.CoreMLJNI;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
     *
     * @param packagePath The path to the *.mlpackage directory.
     * @param modelsDirectory The root directory where models are stored (to find labels file).
     * @throws IllegalArgumentException If the package is invalid or labels cannot be found.
     * @throws IOException If there's an error reading files.
     */
    public CoreMLPackageModel(Path packagePath, File modelsDirectory)
            throws IllegalArgumentException, IOException {
        this.packagePath = packagePath;

        // 1. Verify packagePath is a directory ending with .mlpackage
        if (!Files.isDirectory(packagePath) || !packagePath.getFileName().toString().endsWith(".mlpackage")) {
            throw new IllegalArgumentException("Invalid package path: " + packagePath + ". Must be a directory ending with .mlpackage");
        }

        // 2. Parse package name components
        String packageName = packagePath.getFileName().toString();
        String[] parts;
        try {
             parts = parsePackageName(packageName);
        } catch (IllegalArgumentException e) {
             throw new IllegalArgumentException("Failed to parse package name: " + packageName, e);
        }

        // 3. Determine version and input size
        try {
             this.version = CoreMLFileModel.getModelVersion(parts[3]); // Reuse logic from CoreMLFileModel
             int width = Integer.parseInt(parts[1]);
             int height = Integer.parseInt(parts[2]);
             this.inputSize = new Size(width, height);
        } catch (IllegalArgumentException | IndexOutOfBoundsException e) {
             throw new IllegalArgumentException("Failed to determine version/size from package name parts: " + packageName, e);
        }

        // 4. Find and read the corresponding labels file
        String baseName = packageName.replace(".mlpackage", "");
        Path labelPath = modelsDirectory.toPath().resolve(baseName + "-labels.txt");
        logger.info("Looking for labels file at: " + labelPath);
        if (Files.exists(labelPath)) {
            try {
                 this.labels = Files.readAllLines(labelPath);
                 logger.info("Successfully loaded labels for " + packageName);
            } catch (IOException e) {
                 throw new IOException("Failed to read labels file: " + labelPath, e);
            }
        } else {
            throw new IOException("Could not find labels file: " + labelPath);
        }
    }

    @Override
    public String getName() {
        // Return the package directory name
        return packagePath.getFileName().toString();
    }

    @Override
    public ObjectDetector load() {
        // TODO: Implement loading logic for CoreMLObjectDetector using the package path
        // This might require changes in CoreMLObjectDetector or CoreMLJNI
        logger.warn("load() not implemented for CoreMLPackageModel");
        // Placeholder - Needs adaptation for packages
         // return new CoreMLObjectDetector(this, inputSize); // 'this' might need adjustment, inputSize needs value
         return null; // Or throw exception
    }

    /**
     * Check naming conventions for *.mlpackage.zip files and their corresponding labels files.
     *
     * @param modelZipFileName The name of the uploaded zip file (e.g., my_model-640-640-yolov8.mlpackage.zip).
     * @param labelsFileName The name of the labels file (e.g., my_model-640-640-yolov8-labels.txt).
     * @throws IllegalArgumentException If the names are invalid or don't match.
     */
    public static void verifyNames(String modelZipFileName, String labelsFileName)
            throws IllegalArgumentException {
        // Implement pattern matching and validation for zip/txt names
        // Pattern should look for name-width-height-version.mlpackage.zip
        // and name-width-height-version-labels.txt
        // Reuse patterns from CoreMLFileModel but adapt for .mlpackage.zip
        // Pattern for zip: name-width-height-version.mlpackage.zip
        // Need double backslash for literal dot in Java string regex
        Pattern zipPattern = Pattern.compile("^([a-zA-Z0-9._]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.mlpackage\\.zip$");
        // Pattern for txt: name-width-height-version-labels.txt
        Pattern labelsPattern = Pattern.compile("^([a-zA-Z0-9._]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)-labels\\.txt$");

        // check null
        if (modelZipFileName == null || labelsFileName == null) {
            throw new IllegalArgumentException("Model zip name and labels name cannot be null");
        }

        Matcher zipMatcher = zipPattern.matcher(modelZipFileName);
        Matcher labelsMatcher = labelsPattern.matcher(labelsFileName);

        logger.debug("Model zip name: " + modelZipFileName);
        logger.debug("Labels name: " + labelsFileName);

        if (!zipMatcher.matches()) {
            logger.debug("Model zip name does not match pattern: " + zipPattern.pattern());
        }
        if (!labelsMatcher.matches()) {
            logger.debug("Labels name does not match pattern: " + labelsPattern.pattern());
        }

        if (!zipMatcher.matches() || !labelsMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model zip/labels must follow naming convention: name-width-height-version.mlpackage.zip / name-width-height-version-labels.txt");
        }

        // Check if all captured groups match between the zip and txt file names
        if (!zipMatcher.group(1).equals(labelsMatcher.group(1)) // name
                || !zipMatcher.group(2).equals(labelsMatcher.group(2)) // width
                || !zipMatcher.group(3).equals(labelsMatcher.group(3)) // height
                || !zipMatcher.group(4).equals(labelsMatcher.group(4))) { // version string
            throw new IllegalArgumentException("Model zip name and labels name parts must match.");
        }

        // logger.warn("verifyNames() not implemented for CoreMLPackageModel");
        // Example (needs proper regex):
        // Pattern zipPattern = Pattern.compile("^(.*)\.mlpackage\.zip$");
        // Pattern txtPattern = Pattern.compile("^(.*)-labels\.txt$");
        // Matcher zipMatcher = zipPattern.matcher(modelZipFileName);
        // Matcher txtMatcher = txtPattern.matcher(labelsFileName);
        // if (!zipMatcher.matches() || !txtMatcher.matches() || !zipMatcher.group(1).equals(txtMatcher.group(1))) {
        //    throw new IllegalArgumentException("Invalid naming or mismatch...");
        // }
    }

    // Optional: Add a static parsePackageName method similar to CoreMLFileModel.parseModelName
    // Pattern for package name: name-width-height-version.mlpackage
    private static Pattern packagePattern =
            Pattern.compile("^([a-zA-Z0-9._]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.mlpackage$");

    /**
     * Parse CoreML package name and return the name, width, height, and model type string.
     *
     * @param packageName the name of the package directory (e.g., my_model-640-640-yolov8.mlpackage)
     * @throws IllegalArgumentException if the package name does not follow the naming convention
     * @return an array containing the name, width, height, and model type string
     */
    public static String[] parsePackageName(String packageName)
        throws IllegalArgumentException {
        Matcher packageMatcher = packagePattern.matcher(packageName);

        if (!packageMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Package name must follow naming convention: name-width-height-version.mlpackage. Got: " + packageName);
        }

        return new String[] {
            packageMatcher.group(1), packageMatcher.group(2), packageMatcher.group(3), packageMatcher.group(4)
        };
    }
} 