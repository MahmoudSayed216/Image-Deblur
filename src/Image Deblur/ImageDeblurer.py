class ImageDeblurer:
    DeepDeblur = 0b1
    DeblurGan1 = 0b10
    DeblurGan2 = 0b100
    Restormer  = 0b1000
    def __init__(self, required_models):
        self.load_models(required_models)
        self.models = {self.DeepDeblur: None, self.DeblurGan1: None, self.DeblurGan2: None, self.Restormer: None}
        self.transforms = {self.DeepDeblur: None, self.DeblurGan1: None, self.DeblurGan2: None, self.Restormer: None}

    def load_models(self, required_models) ->None:
        importedModel1 = None
        importedModel2 = None
        importedModel3 = None
        importedModel4 = None

        importedTransforms1 = None
        importedTransforms2 = None
        importedTransforms3 = None
        importedTransforms4 = None

        if required_models & self.DeepDeblur == self.DeepDeblur:
            self.models[self.DeepDeblur] = importedModel1
            self.transforms[self.DeepDeblur] = importedTransforms1
            
        elif required_models & self.DeblurGan1 == self.DeblurGan1:
            self.models[self.DeblurGan1] = importedModel2
            self.transforms[self.DeblurGan1] = importedTransforms2
        elif required_models & self.DeblurGan2 == self.DeblurGan2:
            self.models[self.DeblurGan2] = importedModel3
            self.transforms[self.DeblurGan2] = importedTransforms3
        elif required_models & self.Restormer == self.Restormer:
            self.models[self.Restormer] = importedModel4
            self.transforms[self.Restormer] = importedTransforms4

    def set_models(self, required_models) -> None:
        pass

    def infer(self, image_path, transforms) -> dict:
        output = {self.DeepDeblur: None, self.DeblurGan1: None, self.DeblurGan2: None, self.Restormer: None}

        for model_name in  [self.DeepDeblur, self.DeblurGan1, self.DeblurGan2, self.Restormer]:
            transforms, model = self.transforms[model_name], self.models[model_name] 
            #TODO: Read Image
            #TODO: Pass it to transforms
            #TODO: Pass it to model
            #TODO: Save it to the output dict

        return output