﻿﻿// This file was auto-generated by ML.NET Model Builder. 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace ImageClassificationCodeGen
{
    public partial class ImageClassification
    {
        public static ITransformer RetrainPipeline(MLContext context, IDataView trainData)
        {
            var pipeline = BuildPipeline(context);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
       
        public class IInput
        {
            [VectorType(224*224*3)]
            public byte[] ImageSource_featurized { get; set; }
        }

        public class IOutput
        {
            [VectorType()]
            public byte[] ImageSource_featurized { get; set; }
        }

        [CustomMappingFactoryAttribute("VectorMapping")]
        private class VectorMappingCustomAction : CustomMappingFactory<IInput,
            IOutput>
        {
            public static void CustomAction(IInput input, IOutput output) => output.ImageSource_featurized = input.ImageSource_featurized;

            public override Action<IInput, IOutput> GetMapping() => CustomAction;
        }


        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {

            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(@"Label", @"Label")
                                    .Append(mlContext.Transforms.ResizeImages("ImageSource_featurized", 224, 224, "ImageSource"))
                                    .Append(mlContext.Transforms.ExtractPixels("ImageSource_featurized",outputAsFloatArray:false))
                                    .Append(mlContext.Transforms.CustomMapping(new VectorMappingCustomAction().GetMapping(), contractName: "VectorMapping"))
                                    .Append(mlContext.Transforms.CopyColumns(@"Features", @"ImageSource_featurized"))
                                    .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName:@"Label"))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

            mlContext.ComponentCatalog.RegisterAssembly(typeof(VectorMappingCustomAction).Assembly);

            return pipeline;
        }
    }
}
