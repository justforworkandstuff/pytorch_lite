#import "PytorchLitePlugin.h"
#import "pigeon.h"
#import "PrePostProcessor.h"
//#import "TorchModule.h"
#import "helpers/UIImageExtension.h"
#import <LibTorch/LibTorch.h>
// #import <Libtorch-Lite/Libtorch-Lite.h>


@interface PytorchLitePlugin () <ModelApi>

@property (nonatomic, assign) std::vector<torch::jit::Module*> modulesVector;
@property (nonatomic, strong) NSMutableArray<PrePostProcessor *> *prePostProcessors;

@end

@implementation PytorchLitePlugin

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
    PytorchLitePlugin* instance = [[PytorchLitePlugin alloc] init];
    SetUpModelApi(registrar.messenger, instance);
    instance.prePostProcessors = [NSMutableArray array];
}

- (void)getPredictionCustomIndex:(NSInteger)index input:(NSArray<NSNumber *> *)input shape:(NSArray<NSNumber *> *)shape dtype:(NSString *)dtype completion:(void (^)(NSArray<id> *_Nullable, FlutterError *_Nullable))completion {
    // Implement custom prediction logic here based on 'input', 'shape', and 'dtype'.
    // This is a placeholder, replace with your actual implementation.
    completion(nil, nil);
}


- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer withWidth:(int)width andHeight:(int)height atIndex:(NSInteger)moduleIndex isObjectDetection:(BOOL)isObjectDetection objectDetectionType:(NSInteger)objectDetectionType isTupleOutput:(NSNumber *)isTupleOutput tupleIndex:(NSNumber *)tupleIndex {
    try {
        torch::jit::Module* module = _modulesVector[moduleIndex];
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, height, width}, at::kFloat);

        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        at::Tensor outputTensor;

        if (isObjectDetection) {
            if (objectDetectionType == 0) { // YOLO
                torch::jit::IValue outputTuple = module->forward({tensor}).toTuple();
                outputTensor = outputTuple.toTuple()->elements()[0].toTensor();
            } else { // SSD & other single output models
                outputTensor = module->forward({tensor}).toTensor();
            }
        } else { // Classification, Segmentation, etc.
            /* ----------------------------- ORIGINAL CODE ------------------------- */
            // outputTensor = module->forward({tensor}).toTensor();

            /* ----------------------------- AGMO CUSTOM CODE ------------------------- */
            if (isTupleOutput.boolValue) {
                outputTensor = module->forward({tensor}).toTuple()->elements()[tupleIndex.integerValue].toTensor();
            } else {
                outputTensor = module->forward({tensor}).toTensor();
            }
        }

        float *floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }

        int prod = 1;
        for (int i = 0; i < outputTensor.sizes().size(); i++) {
            prod *= outputTensor.sizes().data()[i];
        }

        NSMutableArray<NSNumber*>* results = [[NSMutableArray<NSNumber*> alloc] init];
        for (int i = 0; i < prod; i++) {
            [results addObject: @(floatBuffer[i])];
        }

        return [results copy];

    } catch (const std::exception& e) {
        NSLog(@"%s", e.what());
        return nil;
    }
}




- (void)loadModelModelPath:(NSString *)modelPath numberOfClasses:(nullable NSNumber *)numberOfClasses imageWidth:(nullable NSNumber *)imageWidth imageHeight:(nullable NSNumber *)imageHeight objectDetectionModelType:(nullable NSNumber *)objectDetectionModelType completion:(void (^)(NSNumber *_Nullable, FlutterError *_Nullable))completion{
    NSInteger i = -1;
    try {
        torch::jit::Module *module = new torch::jit::Module(torch::jit::load(modelPath.UTF8String));
        _modulesVector.push_back(module);

if (numberOfClasses != nil && imageWidth != nil && imageHeight != nil) {
            [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithNumberOfClasses:numberOfClasses.integerValue imageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue objectDetectionModelType:objectDetectionModelType.integerValue]];
        } else {
            if (imageWidth != nil && imageHeight != nil) {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithImageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue]];
            } else {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] init]];
            }
        }
        i = _modulesVector.size() - 1;
        completion(@(i), nil);

    } catch (const std::exception& e) {
         NSLog(@"%@ is not a proper model: %s", modelPath, e.what());
        FlutterError *error = [FlutterError errorWithCode:@"ModelLoadingError" message:[NSString stringWithFormat:@"%@ is not a proper model", modelPath] details:@(e.what())];
        completion(nil, error);
    }
}



- (void)getImagePredictionListIndex:(NSInteger)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList mean:(NSArray<NSNumber *> *)mean std:(NSArray<NSNumber *> *)std isTupleOutput:(nonnull NSNumber *)isTupleOutput tupleIndex:(nonnull NSNumber *)tupleIndex completion:(void (^)(NSArray<NSNumber *> *_Nullable, FlutterError *_Nullable))completion {
    UIImage *bitmap = nil;
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];

    if (imageData) {
        bitmap = [UIImage imageWithData:imageData.data];
    } else {
        FlutterStandardTypedData *typedData = imageBytesList[0];
        uint8_t* in = (uint8_t*)[[typedData data] bytes];

        bitmap = [UIImage imageWithData:typedData.data];
    }
    bitmap = [UIImageExtension resize:bitmap toWidth:prePostProcessor.mImageWidth toHeight:prePostProcessor.mImageHeight];

    float* input = [UIImageExtension normalize:bitmap withMean:mean withSTD:std];
    NSArray<NSNumber*> *results = [self predictImage:input withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:FALSE objectDetectionType:0 isTupleOutput:isTupleOutput tupleIndex:tupleIndex];

    if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}



- (void)getImagePredictionListObjectDetectionIndex:(NSInteger)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList minimumScore:(double)minimumScore IOUThreshold:(double)IOUThreshold boxesLimit:(NSInteger)boxesLimit isTupleOutput:(nonnull NSNumber *)isTupleOutput tupleIndex:(nonnull NSNumber *)tupleIndex completion:(void (^)(NSArray<ResultObjectDetection *> *_Nullable, FlutterError *_Nullable))completion {
     UIImage *bitmap = nil;
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
    prePostProcessor.mNmsLimit = boxesLimit;
    prePostProcessor.mScoreThreshold = minimumScore;
    prePostProcessor.mIOUThreshold = IOUThreshold;

    if (imageData) {
        bitmap = [UIImage imageWithData:imageData.data];
    } else {
        FlutterStandardTypedData *typedData = imageBytesList[0];
        uint8_t* in = (uint8_t*)[[typedData data] bytes];
        bitmap = [UIImage imageWithData:typedData.data];
    }
    bitmap = [UIImageExtension resize:bitmap toWidth:prePostProcessor.mImageWidth toHeight:prePostProcessor.mImageHeight];
    float* input = [UIImageExtension normalize:bitmap withMean:prePostProcessor.NO_MEAN_RGB withSTD:prePostProcessor.NO_STD_RGB];
    NSArray<NSNumber*> *rawOutputs = [self predictImage:input withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:TRUE objectDetectionType:prePostProcessor.mObjectDetectionModelType isTupleOutput:isTupleOutput tupleIndex:tupleIndex];

    NSMutableArray<ResultObjectDetection*> *results = [prePostProcessor outputsToNMSPredictions:rawOutputs];
 if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}


- (void)getRawImagePredictionListIndex:(NSInteger)index imageData:(FlutterStandardTypedData *)imageData isTupleOutput:(nonnull NSNumber *)isTupleOutput tupleIndex:(nonnull NSNumber *)tupleIndex completion:(void (^)(NSArray<NSNumber *> *_Nullable, FlutterError *_Nullable))completion {
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
    /* ----------------------------- ORIGINAL CODE ------------------------- */
    // NSArray<NSNumber*> *results = [self predictImage:(float *)[imageData.data bytes] withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:FALSE objectDetectionType:0];
    
    // if (results) {
    //     completion(results, nil);
    // } else {
    //     FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
    //     completion(nil, error);
    // }

    /* ----------------------------- AGMO CUSTOM CODE ------------------------- */
    torch::jit::Module *module = _modulesVector[index];

    @try {
        NSUInteger imageWidth = prePostProcessor.mImageWidth;
        NSUInteger imageHeight = prePostProcessor.mImageHeight;
        NSUInteger bufferSize = 3 * imageWidth * imageHeight;
        // FloatBuffer floatBuffer = (FloatBuffer)malloc(bufferSize * sizeof(float));
        float *floatBuffer = (float *)malloc(bufferSize * sizeof(float));

        if (!floatBuffer) {
            FlutterError *error = [FlutterError errorWithCode:@"MEMORY_ALLOCATION_FAILED" 
                                                      message:@"Failed to allocate memory for floatBuffer"
                                                      details:nil];
            completion(nil, error);
            return;
        }

        // ByteBuffer byteBuffer = ByteBuffer.wrap(imageData.data.bytes, imageData.data.length);
        // byteBuffer.order(ByteOrder.nativeOrder());
        // FloatBuffer tempFloatBuffer = byteBuffer.asFloatBuffer();
        // memcpy(floatBuffer, tempFloatBuffer, bufferSize * sizeof(float));

        memcpy(floatBuffer, imageData.data.bytes, MIN(imageData.data.length, bufferSize * sizeof(float)));

        // at::Tensor *imageInputTensor = [at::Tensor fromBlobWithFloatBuffer:floatBuffer dims:@[@1, @3, @(imageHeight), @(imageWidth)]];
        at::Tensor imageInputTensor = torch::from_blob(floatBuffer, {1, 3, static_cast<long long>(imageHeight), static_cast<long long>(imageWidth)}, torch::kFloat);

        // Free float buffer after tensor creation
        free(floatBuffer);

        at::Tensor imageOutputTensor;
        if ([isTupleOutput boolValue]) {
            imageOutputTensor = module->forward({imageInputTensor}).toTuple()->elements()[tupleIndex.intValue].toTensor();
        } else {
            imageOutputTensor = module->forward({imageInputTensor}).toTensor();
        }

        // Assuming that 'imageOutputTensor' is correctly assigned before this

        NSMutableArray<NSNumber *> *doubleArray = [NSMutableArray array];

        if (imageOutputTensor.dtype() == at::kQUInt8) {
            // Handle unsigned 8-bit integers
            uint8_t *tensorData = imageOutputTensor.data_ptr<uint8_t>(); // Get raw unsigned byte data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert byte to NSNumber
            }
        } else if (imageOutputTensor.dtype() == at::kQInt8) {
            // Handle signed 8-bit integers
            int8_t *tensorData = imageOutputTensor.data_ptr<int8_t>(); // Get raw signed byte data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert byte to NSNumber
            }
        } else if (imageOutputTensor.dtype() == at::kInt) {
            // Handle int32
            int *tensorData = imageOutputTensor.data_ptr<int>(); // Get raw int32 data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert int to NSNumber
            }
        } else if (imageOutputTensor.dtype() == at::kFloat) {
            // Handle float32
            float *tensorData = imageOutputTensor.data_ptr<float>(); // Get raw float data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert float to NSNumber
            }
        } else if (imageOutputTensor.dtype() == at::kLong) {
            // Handle int64 (Long)
            int64_t *tensorData = imageOutputTensor.data_ptr<int64_t>(); // Get raw int64 data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert int64 to NSNumber
            }
        } else if (imageOutputTensor.dtype() == at::kDouble) {
            // Handle float64
            double *tensorData = imageOutputTensor.data_ptr<double>(); // Get raw double data
            for (int i = 0; i < imageOutputTensor.numel(); i++) {
                [doubleArray addObject:@(tensorData[i])]; // Convert double to NSNumber
            }
        } else {
            NSLog(@"Unsupported tensor data type");
            completion(nil, nil);
            return;
        }

        completion(doubleArray, nil);
        // NSMutableArray<NSNumber *> *doubleArray = [NSMutableArray array];
        // switch (imageOutputTensor.dtype) {
        //     case TENSOR_DTYPE_UINT8: {
        //         // NSData *byteArray = [imageOutputTensor getDataAsUnsignedByteArray];
        //         // NSUInteger length = byteArray.length;
        //         // for (NSUInteger i = 0; i < length; i++) {
        //         //     uint8_t byteValue;
        //         //     [byteArray getBytes:&byteValue range:NSMakeRange(i, 1)];
        //         //     [doubleArray addObject:@((double)byteValue)];
        //         // }

        //         NSMutableArray<NSNumber *> *byteArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         uint8_t *tensorData = imageOutputTensor.data_ptr<uint8_t>(); // Get raw unsigned byte data from tensor

        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [byteArray addObject:@(tensorData[i])]; // Convert each byte to NSNumber
        //         }

        //         [doubleArray addObjectsFromArray:byteArray]; // Add all elements to doubleArray
        //     }
        //         break;
        //     case TENSOR_DTYPE_INT8: {
        //         // NSData *byteArray = [imageOutputTensor getDataAsByteArray];
        //         // NSUInteger length = byteArray.length;
        //         // for (NSUInteger i = 0; i < length; i++) {
        //         //     int8_t byteValue;
        //         //     [byteArray getBytes:&byteValue range:NSMakeRange(i, 1)];
        //         //     [doubleArray addObject:@((double)byteValue)];
        //         // }

        //         NSMutableArray<NSNumber *> *byteArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         int8_t *tensorData = imageOutputTensor.data_ptr<int8_t>(); // Get raw signed byte data from tensor

        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [byteArray addObject:@(tensorData[i])]; // Convert each byte to NSNumber
        //         }

        //         [doubleArray addObjectsFromArray:byteArray]; // Add all elements to doubleArray
        //     }
        //         break;
        //     case TENSOR_DTYPE_INT32: {
        //         // NSArray<NSNumber *> *intArray = [imageOutputTensor getDataAsIntArray];
        //         // for (NSNumber *num in intArray) {
        //         //     [doubleArray addObject:@((double)[num intValue])];
        //         // }

        //         NSMutableArray<NSNumber *> *intArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         int *tensorData = imageOutputTensor.data_ptr<int>(); // Get raw int data from tensor

        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [intArray addObject:@(tensorData[i])]; // Convert int to NSNumber
        //         }
        //         [doubleArray addObjectsFromArray:intArray]; // Add all elements to doubleArray
        //     }
        //         break;
        //     case TENSOR_DTYPE_FLOAT32: {
        //         // NSArray<NSNumber *> *floatArray = [imageOutputTensor getDataAsFloatArray];
        //         // for (NSNumber *num in floatArray) {
        //         //     [doubleArray addObject:@((double)[num floatValue])];
        //         // }

        //         NSMutableArray<NSNumber *> *floatArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         float *tensorData = imageOutputTensor.data_ptr<float>(); // Get raw float data from tensor

        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [floatArray addObject:@(tensorData[i])]; // Convert float to NSNumber
        //         }
        //         [doubleArray addObjectsFromArray:floatArray]; // Add all elements to doubleArray
        //     }
        //         break;
        //     case TENSOR_DTYPE_INT64: {
        //         // NSArray<NSNumber *> *longArray = [imageOutputTensor getDataAsLongArray];
        //         // for (NSNumber *num in longArray) {
        //         //     [doubleArray addObject:@((double)[num longValue])];
        //         // }

        //         NSMutableArray<NSNumber *> *longArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         int64_t *tensorData = imageOutputTensor.data_ptr<int64_t>(); // Get raw int64 data from tensor

        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [longArray addObject:@(tensorData[i])];
        //         }
        //         [doubleArray addObjectsFromArray:longArray];
        //     }
        //         break;
        //     case TENSOR_DTYPE_FLOAT64: {
        //         // NSArray<NSNumber *> *rawDoubleArray = [imageOutputTensor getDataAsDoubleArray];
        //         // [doubleArray addObjectsFromArray:rawDoubleArray];

        //         NSMutableArray<NSNumber *> *doubleArray = [NSMutableArray arrayWithCapacity:imageOutputTensor.numel()];
        //         double *tensorData = imageOutputTensor.data_ptr<double>(); // Extract raw data
        //         for (int i = 0; i < imageOutputTensor.numel(); i++) {
        //             [doubleArray addObject:@(tensorData[i])];
        //         }
        //     }
        //         break;
        //     default:
        //         NSLog(@"Unsupported tensor data type");
        //         completion(nil, nil);
        //         return;
        // }

        // completion(doubleArray, nil);
    } @catch (NSException *e) {
        NSLog(@"Error classifying image: %@", e);
        FlutterError *error = [FlutterError errorWithCode:@"EXCEPTION" message:@"An error occurred while classifying the image" details:e.reason];
        completion(nil, error);
    }
}


- (void)getRawImagePredictionListObjectDetectionIndex:(NSInteger)index imageData:(FlutterStandardTypedData *)imageData minimumScore:(double)minimumScore IOUThreshold:(double)IOUThreshold boxesLimit:(NSInteger)boxesLimit isTupleOutput:(nonnull NSNumber *)isTupleOutput tupleIndex:(nonnull NSNumber *)tupleIndex completion:(void (^)(NSArray<ResultObjectDetection *> *_Nullable, FlutterError *_Nullable))completion {
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
    prePostProcessor.mNmsLimit = boxesLimit;
    prePostProcessor.mScoreThreshold = minimumScore;
    prePostProcessor.mIOUThreshold = IOUThreshold;

    NSArray<NSNumber*> *rawOutputs = [self predictImage:(float *)[imageData.data bytes] withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:TRUE objectDetectionType:prePostProcessor.mObjectDetectionModelType isTupleOutput:isTupleOutput tupleIndex:tupleIndex];

    NSMutableArray<ResultObjectDetection*> *results = [prePostProcessor outputsToNMSPredictions:rawOutputs];
    if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}


- (void)dealloc {
    for (torch::jit::Module* module : _modulesVector) {
        delete module;
    }
    _modulesVector.clear();
}


@end
