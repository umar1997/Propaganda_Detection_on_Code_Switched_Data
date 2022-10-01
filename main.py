from preprocess import Dataset_Processing


if __name__ == '__main__':

    paths = {
        "Meme_Data": "./Meme_Data/",
        "Meme_Data_Train_Json": "./Meme_Data/training_set_.json",
        "Meme_Data_Val_Json": "./Meme_Data/dev_set_.json",
        "Meme_Data_Test_Json": "./Meme_Data/test_set_.json",
        "Meme_Data_Train":"./Meme_Data/training_set_.csv",
        "Meme_Data_Val":"./Meme_Data/dev_set_.csv",
        "Meme_Data_Test":"./Meme_Data/test_set_.csv",
    }

    ##################################################  PREPROCESSING
    # dataProcessing = Dataset_Processing(paths)
    # data_df = dataProcessing.run()
