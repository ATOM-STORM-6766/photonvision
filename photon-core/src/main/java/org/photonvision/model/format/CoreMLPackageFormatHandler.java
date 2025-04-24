package org.photonvision.model.format;

import io.javalin.http.UploadedFile;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.coreml.CoreMLJNI;
import org.photonvision.model.vision.CoreMLModel;
import org.photonvision.model.vision.Model;


public class CoreMLPackageFormatHandler implements ModelFormatHandler {

    private static final Logger logger = new Logger(CoreMLPackageFormatHandler.class, LogGroup.Config);
    private static final String BACKEND_NAME = "COREML_PACKAGE";
    private static final String PRIMARY_EXTENSION = ".mlpackage";
    private static final String UPLOAD_EXTENSION = ".zip";
    private static final Class<? extends Model> MODEL_CLASS = CoreMLModel.class;

    // Naming convention patterns
    // Zip: name-width-height-version.mlpackage.zip
    private static final Pattern zipPattern =
            Pattern.compile("^([a-zA-Z0-9._-]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.mlpackage\\.zip$");
    // Labels: name-width-height-version-labels.txt
    private static final Pattern labelsPattern =
            Pattern.compile("^([a-zA-Z0-9._-]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)-labels\\.txt$");
    // Package Dir: name-width-height-version.mlpackage
     private static final Pattern packageDirPattern =
             Pattern.compile("^([a-zA-Z0-9._-]+)-(\\d+)-(\\d+)-(yolov(?:5|8|11)[nsmlx]*)\\.mlpackage$");


    // Helper class to hold parsed name components
    private static class ParsedModelInfo {
        final String baseName;
        final int width;
        final int height;
        final String versionString;
        final CoreMLJNI.ModelVersion version;
        final Size inputSize;
        final String expectedPackageDirName; // e.g., name-w-h-v.mlpackage

        ParsedModelInfo(String baseName, int width, int height, String versionString) {
            this.baseName = baseName;
            this.width = width;
            this.height = height;
            this.versionString = versionString;
            this.version = parseVersionString(versionString); // Determine enum version
            this.inputSize = new Size(width, height);
            this.expectedPackageDirName = baseName + "-" + width + "-" + height + "-" + versionString + PRIMARY_EXTENSION;
        }
    }


    @Override
    public String getBackendName() {
        return BACKEND_NAME;
    }

    @Override
    public String getUploadAcceptType() {
        return UPLOAD_EXTENSION;
    }

    @Override
    public Class<? extends Model> getModelClass() {
        return MODEL_CLASS;
    }

    @Override
    public Info getInfo() {
        return new Info(getBackendName(), getUploadAcceptType());
    }

    @Override
    public boolean supportsPath(Path path) {
        if (path == null) return false;
        // Check if it's a directory and matches the package naming convention
        return Files.isDirectory(path) && packageDirPattern.matcher(path.getFileName().toString()).matches();
    }

    @Override
    public boolean supportsUpload(String modelZipFileName, String labelsFileName) {
         if (modelZipFileName == null || labelsFileName == null) return false;
         try {
             verifyNames(modelZipFileName, labelsFileName); // Use the stricter verifyNames
             return true;
         } catch (IllegalArgumentException e) {
             return false;
         }
    }

    @Override
    public Optional<String> validateUpload(UploadedFile modelZipFile, UploadedFile labelsFile) {
        String modelZipName = modelZipFile.filename();
        String labelsName = labelsFile.filename();

        // Basic extension checks
        if (!modelZipName.toLowerCase().endsWith(UPLOAD_EXTENSION)) {
             return Optional.of("Invalid model file type. Expected '" + UPLOAD_EXTENSION + "' but got '" + modelZipName + "'");
         }
         if (!labelsName.toLowerCase().endsWith(".txt")) {
             return Optional.of("Invalid labels file type. Expected '.txt' but got '" + labelsName + "'");
         }

         // Deeper validation using naming convention
         try {
             verifyNames(modelZipName, labelsName);
         } catch (IllegalArgumentException e) {
             return Optional.of("Invalid file naming convention: " + e.getMessage());
         }

        return Optional.empty();
    }

    @Override
    public void verifyNames(String modelZipFileName, String labelsFileName) throws IllegalArgumentException {
        if (modelZipFileName == null || labelsFileName == null) {
            throw new IllegalArgumentException("Model zip name and labels name cannot be null");
        }

        Matcher zipMatcher = zipPattern.matcher(modelZipFileName);
        Matcher labelsMatcher = labelsPattern.matcher(labelsFileName);

        logger.debug("Verifying CoreML Package names - Zip: " + modelZipFileName + ", Labels: " + labelsFileName);

        if (!zipMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Model zip name '" + modelZipFileName + "' must follow the convention name-width-height-version" + UPLOAD_EXTENSION);
        }
        if (!labelsMatcher.matches()) {
            throw new IllegalArgumentException(
                    "Labels name '" + labelsFileName + "' must follow the convention name-width-height-version-labels.txt");
        }

        // Check if all captured groups match
        if (!zipMatcher.group(1).equals(labelsMatcher.group(1)) // baseName
                || !zipMatcher.group(2).equals(labelsMatcher.group(2)) // width
                || !zipMatcher.group(3).equals(labelsMatcher.group(3)) // height
                || !zipMatcher.group(4).equals(labelsMatcher.group(4))) { // versionString
            throw new IllegalArgumentException(
                "Model zip name ('" + modelZipFileName + ") and labels name ('" + labelsFileName + ") parts must match.");
        }

        // Additionally parse and check numeric parts and version
        try {
            parseZipName(modelZipFileName);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid components in model zip name: " + e.getMessage(), e);
        }
    }

    @Override
    public Model loadFromPath(Path packagePath, File modelsDirectory) throws IOException, IllegalArgumentException {
        String packageDirName = packagePath.getFileName().toString();
        logger.info("Loading CoreML package from path: " + packagePath);

        if (!supportsPath(packagePath)) { // Double check it's a valid package dir
             throw new IllegalArgumentException("Path is not a valid CoreML package directory: " + packagePath);
        }

        // 1. Parse package directory name to get info
        ParsedModelInfo parsedInfo;
        try {
            parsedInfo = parsePackageDirName(packageDirName);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Failed to parse package directory name: " + packageDirName, e);
        }

        // 2. Determine and find the labels file (relative to modelsDirectory)
        String labelsFileName = deriveLabelsName(parsedInfo);
        Path labelsPath = modelsDirectory.toPath().resolve(labelsFileName);

        if (!Files.exists(labelsPath) || !Files.isRegularFile(labelsPath)) {
            throw new IOException("Could not find or access expected labels file: " + labelsPath);
        }

        // 3. Read labels
        List<String> labels;
        try {
            labels = Files.readAllLines(labelsPath);
            logger.debug("Successfully read labels from: " + labelsPath);
        } catch (IOException e) {
            throw new IOException("Failed to read labels file: " + labelsPath, e);
        }

        // 4. Create the CoreMLModel instance
        try {
             // Updated to use unified CoreMLModel with isPackage=true
             return new CoreMLModel(packagePath, true, labels, parsedInfo.version, parsedInfo.inputSize);
        } catch (Exception e) {
             throw new IOException("Failed to instantiate CoreMLModel for " + packageDirName, e);
        }
    }

    @Override
    public void saveUploadedFiles(UploadedFile modelZipFile, UploadedFile labelsFile, File modelsDirectory) throws IOException {
         // Ensure target directory exists
         ensureDirectoryExists(modelsDirectory);

         String modelZipFileName = modelZipFile.filename();
         String labelsFileName = labelsFile.filename();

         // Verify names before saving
         try {
              verifyNames(modelZipFileName, labelsFileName);
         } catch (IllegalArgumentException e) {
              throw new IOException("Uploaded file names are invalid: " + e.getMessage(), e);
         }

         // Parse info to get expected package directory name
         ParsedModelInfo parsedInfo = parseZipName(modelZipFileName); // Already validated by verifyNames

         Path labelsDestPath = modelsDirectory.toPath().resolve(labelsFileName);
         Path zipDestPath = modelsDirectory.toPath().resolve(modelZipFileName); // Temporary path for the zip
         Path packageDestPath = modelsDirectory.toPath().resolve(parsedInfo.expectedPackageDirName);

         logger.info("Saving CoreML Package files to: " + modelsDirectory.getAbsolutePath());
         logger.debug("Target Labels: " + labelsDestPath);
         logger.debug("Target Package Dir: " + packageDestPath);
         logger.debug("Temporary Zip: " + zipDestPath);


         // --- Critical Section: Potential partial state ---
         boolean success = false;
         Path createdPackageDir = null; // Track if directory was created for cleanup

         try {
             // 1. Save labels file
             logger.debug("Saving labels file...");
             try (InputStream in = labelsFile.content(); OutputStream out = Files.newOutputStream(labelsDestPath)) {
                 in.transferTo(out);
             } catch (IOException e) {
                 throw new IOException("Failed to save labels file: " + labelsDestPath, e); // Fail early
             }

             // 2. Save the temporary zip file
             logger.debug("Saving temporary zip file...");
              try (InputStream in = modelZipFile.content()) {
                  Files.copy(in, zipDestPath, StandardCopyOption.REPLACE_EXISTING);
              } catch (IOException e) {
                  cleanupResource(labelsDestPath, "labels file"); // Clean up labels
                  throw new IOException("Failed to save temporary zip file: " + zipDestPath, e);
              }

             // 3. Unzip the package
             logger.debug("Unzipping " + zipDestPath + " to " + packageDestPath + "...");
             // Ensure target package dir doesn't exist or is empty (delete if necessary)
             if (Files.exists(packageDestPath)) {
                 logger.warn("Target package directory " + packageDestPath + " already exists. Deleting it before unzipping.");
                 try {
                    deleteDirectoryRecursively(packageDestPath);
                 } catch (IOException e) {
                    cleanupResource(labelsDestPath, "labels file");
                    cleanupResource(zipDestPath, "temporary zip file");
                    throw new IOException("Failed to delete existing package directory: " + packageDestPath, e);
                 }
             }

             try {
                 unzipFile(zipDestPath, packageDestPath.getParent()); // Unzip into the models directory
                 createdPackageDir = packageDestPath; // Mark directory as created
                 // Verify the unzipped directory has the expected name
                 if (!Files.isDirectory(packageDestPath)) {
                     throw new IOException("Unzipping did not create the expected directory: " + packageDestPath);
                 }
                  logger.info("Successfully unzipped package to: " + packageDestPath);
             } catch (IOException e) {
                 cleanupResource(labelsDestPath, "labels file");
                 cleanupResource(zipDestPath, "temporary zip file");
                 // Clean up potentially partially extracted directory
                 if (createdPackageDir != null && Files.exists(createdPackageDir)) {
                     try { deleteDirectoryRecursively(createdPackageDir); } catch (IOException ignored) {}
                 }
                 throw new IOException("Failed to unzip model package: " + zipDestPath + " -> " + e.getMessage(), e);
             }

             // 4. Delete the temporary zip file
             logger.debug("Deleting temporary zip file: " + zipDestPath);
             try {
                 Files.delete(zipDestPath);
             } catch (IOException e) {
                 // This is not ideal, but the main operation succeeded. Log a warning.
                 logger.warn("Failed to delete temporary zip file after successful unzip: " + zipDestPath + " Error: " + e.getMessage());
             }

             success = true;
             logger.info("Successfully saved CoreML package " + parsedInfo.expectedPackageDirName + " and labels " + labelsFileName);

         } finally {
             // Ensure cleanup if something unexpected happened after zip creation but before success
             if (!success) {
                 logger.error("CoreML Package save failed. Attempting cleanup...");
                 cleanupResource(labelsDestPath, "labels file");
                 cleanupResource(zipDestPath, "temporary zip file");
                 if (createdPackageDir != null && Files.exists(createdPackageDir)) {
                    logger.debug("Cleaning up created package directory: " + createdPackageDir);
                     try { deleteDirectoryRecursively(createdPackageDir); } catch (IOException ignored) {
                         logger.error("Failed to cleanup package directory: " + createdPackageDir, ignored);
                     }
                 }
             }
         }
    }


    private void ensureDirectoryExists(File directory) throws IOException {
         if (!directory.exists()) {
             if (!directory.mkdirs()) {
                 throw new IOException("Failed to create directory: " + directory.getAbsolutePath());
             }
         }
         if (!directory.isDirectory()) {
             throw new IOException("Path exists but is not a directory: " + directory.getAbsolutePath());
         }
    }

    private void cleanupResource(Path path, String description) {
        if (path != null && Files.exists(path)) {
            try {
                if (Files.isDirectory(path)) {
                     deleteDirectoryRecursively(path);
                     logger.warn("Cleaned up partially created/extracted directory: " + path);
                } else {
                     Files.delete(path);
                     logger.warn("Cleaned up file: " + path + " (" + description + ")");
                }
            } catch (IOException ex) {
                logger.error("Failed to clean up resource: " + path + " (" + description + ")", ex);
            }
        }
    }

     /** Deletes a directory and all its contents recursively. */
    private void deleteDirectoryRecursively(Path path) throws IOException {
        // From https://stackoverflow.com/a/37757112/
        Files.walk(path)
            .sorted(Comparator.reverseOrder())
            .map(Path::toFile)
            // .peek(System.out::println) // uncomment to see which files are deleted
            .forEach(File::delete);
    }


    /**
     * Unzips a zip file to a specified destination directory.
     * Creates the destination directory if it doesn't exist.
     *
     * @param zipFilePath Path to the zip file.
     * @param destDirectory Path to the destination directory.
     * @throws IOException If an I/O error occurs.
     */
    private void unzipFile(Path zipFilePath, Path destDirectory) throws IOException {
        ensureDirectoryExists(destDirectory.toFile()); // Ensure parent directory exists

        byte[] buffer = new byte[1024];
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipFilePath))) {
            ZipEntry zipEntry = zis.getNextEntry();
            while (zipEntry != null) {
                if (isMacOSMetaData(zipEntry.getName())) {
                    logger.debug("Skipping macOS metadata entry: " + zipEntry.getName());
                    zis.closeEntry();
                    zipEntry = zis.getNextEntry();
                    continue;
                }
                
                Path newPath = zipSlipProtect(zipEntry, destDirectory); // Prevent Zip Slip vulnerability
                if (zipEntry.isDirectory()) {
                    if (!Files.isDirectory(newPath)) {
                        Files.createDirectories(newPath);
                    }
                } else {
                    // Ensure parent directory exists for the file
                    Path parent = newPath.getParent();
                    if (parent != null && !Files.isDirectory(parent)) {
                        Files.createDirectories(parent);
                    }

                    // Write file content
                    try (FileOutputStream fos = new FileOutputStream(newPath.toFile())) {
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                }
                zis.closeEntry();
                zipEntry = zis.getNextEntry();
            }
        }
    }

    /** Helper to prevent Zip Slip vulnerability. */
     private Path zipSlipProtect(ZipEntry zipEntry, Path targetDir) throws IOException {
         // From https://snyk.io/research/zip-slip-vulnerability
         Path targetDirResolved = targetDir.resolve(zipEntry.getName());

         // Make sure normalized path doesn't escape the target directory
         Path normalizePath = targetDirResolved.normalize();
         if (!normalizePath.startsWith(targetDir)) {
             throw new IOException("Bad zip entry: " + zipEntry.getName());
         }
         return normalizePath;
     }


    /**
     * Parse CoreML zip file name.
     */
    private ParsedModelInfo parseZipName(String zipFileName) throws IllegalArgumentException {
        Matcher zipMatcher = zipPattern.matcher(zipFileName);
        if (!zipMatcher.matches()) {
             throw new IllegalArgumentException(
                     "Zip name '" + zipFileName + "' must follow the convention name-width-height-version" + UPLOAD_EXTENSION);
         }
         return parseMatcherGroups(zipMatcher);
    }

     /**
      * Parse CoreML package directory name.
      */
     private ParsedModelInfo parsePackageDirName(String packageDirName) throws IllegalArgumentException {
         Matcher dirMatcher = packageDirPattern.matcher(packageDirName);
         if (!dirMatcher.matches()) {
             throw new IllegalArgumentException(
                     "Package directory name '" + packageDirName + "' must follow the convention name-width-height-version" + PRIMARY_EXTENSION);
         }
         return parseMatcherGroups(dirMatcher);
     }

     /** Common parsing logic for matched groups. */
     private ParsedModelInfo parseMatcherGroups(Matcher matcher) throws IllegalArgumentException{
          try {
             String baseName = matcher.group(1);
             int width = Integer.parseInt(matcher.group(2));
             int height = Integer.parseInt(matcher.group(3));
             String versionString = matcher.group(4);

             if (width <= 0 || height <= 0) {
                 throw new IllegalArgumentException("Width and height must be positive integers.");
             }

             // Validate version string via parseVersionString (duplicated logic as requested)
             parseVersionString(versionString); // Throws if invalid

             return new ParsedModelInfo(baseName, width, height, versionString);
         } catch (NumberFormatException e) {
             throw new IllegalArgumentException("Invalid width/height number in name", e);
         } catch (IllegalArgumentException e) {
              throw new IllegalArgumentException("Invalid version string in name: " + e.getMessage(), e);
         }
     }

    /**
     * Determines the model version enum based on the version string.
     * NOTE: This logic is intentionally duplicated from CoreMLFileFormatHandler per user request.
     */
    private static CoreMLJNI.ModelVersion parseVersionString(String versionString) throws IllegalArgumentException {
        if (versionString.startsWith("yolov5")) {
            return CoreMLJNI.ModelVersion.YOLO_V5;
        } else if (versionString.startsWith("yolov8")) {
            return CoreMLJNI.ModelVersion.YOLO_V8;
        } else if (versionString.startsWith("yolov11")) {
            return CoreMLJNI.ModelVersion.YOLO_V11;
        } else {
            throw new IllegalArgumentException("Unknown model version string: " + versionString);
        }
    }

    /**
     * Derive the labels filename from parsed model info.
     */
    private String deriveLabelsName(ParsedModelInfo parsedInfo) {
        return parsedInfo.baseName + "-" + parsedInfo.width + "-" + parsedInfo.height + "-" + parsedInfo.versionString + "-labels.txt";
    }

    /**
     * Check if the entry name is a macOS metadata file.
     */
    private boolean isMacOSMetaData(String entryName) {
        return entryName.startsWith("__MACOSX/") ||
               entryName.endsWith("/.DS_Store") ||
               entryName.equals(".DS_Store");
    }

} 