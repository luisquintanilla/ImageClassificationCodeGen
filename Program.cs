// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Data;
using Input = ImageClassificationCodeGen.ImageClassification.ModelInput;
using System.Drawing;
//using SixLabors.ImageSharp;

IEnumerable<Input> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    var files = Directory.GetFiles(folder, "*",
        searchOption: SearchOption.AllDirectories);

    foreach (var file in files)
    {
        if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
            continue;

        var label = Path.GetFileName(file);

        if (useFolderNameAsLabel)
            label = Directory.GetParent(file).Name;
        else
        {
            for (int index = 0; index < label.Length; index++)
            {
                if (!char.IsLetter(label[index]))
                {
                    label = label.Substring(0, index);
                    break;
                }
            }
        }

        yield return new Input()
        {
            ImageSource = new Bitmap(file),
            //ImageSource = Image.Load(file),
            Label = label
        };
    }
}

void TrainModel()
{
    var ctx = new MLContext();

    var inputData = LoadImagesFromDirectory(@"C:\Datasets\Vehicles-Images-Small");

    var idv = ctx.Data.LoadFromEnumerable<Input>(inputData);

    Console.WriteLine("Training model");

    var retrainedModel = ImageClassificationCodeGen.ImageClassification.RetrainPipeline(ctx, idv);

    ctx.Model.Save(retrainedModel, idv.Schema, "ImageClassification-Retrained.zip");

    Console.WriteLine("Done training");
}

//TrainModel();

void MakePrediction()
{
    var modelInputFile = new Input
    {
        ImageSource = (Bitmap)Bitmap.FromFile(@"C:\Datasets\Vehicles-Images-Small\vehicle\1.png")
        //ImageSource = Image.Load(@"C:\Datasets\Vehicles-Images-Small\vehicle\1.png")
    };

    var modelInputStream = new Input
    {
        ImageSource = (Bitmap)Bitmap.FromStream(File.OpenRead(@"C:\Datasets\Vehicles-Images-Small\vehicle\2.png"))
        //ImageSource = Image.Load(File.OpenRead(@"C:\Datasets\Vehicles-Images-Small\vehicle\2.png"))
    };

    var predictionFile = ImageClassificationCodeGen.ImageClassification.Predict(modelInputFile);
    var predictionStream = ImageClassificationCodeGen.ImageClassification.Predict(modelInputStream);

    Console.WriteLine($"File: {predictionFile.Prediction} | Stream : {predictionStream.Prediction}");
}

MakePrediction();