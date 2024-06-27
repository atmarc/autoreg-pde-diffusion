import os
import copy
from typing import Dict
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler
from torch.nn.utils import prune
import logging
import argparse

from turbpred.model import PredictionModel
from turbpred.logger import Logger
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.trainer import Trainer, Tester
from turbpred.model_diffusion_blocks import Unet, ConvNextBlock



def _prune(model:PredictionModel, prune_type:str, pruning_percentage:float):
    """ Prune Unet model """

    unet = model.modelDecoder
    conv_next_modules = [module for module in unet.modules() if type(module) is ConvNextBlock]
    
    if prune_type == 'L1':
        norm_n = 1
    elif prune_type == 'L2':
        norm_n = 2
    else:
        raise Exception("Wrong prune_type")

    for module in conv_next_modules:
        # We prune modules of ConvNextBlock that are scaled by convnext_mult 
        # module.net[1] --> Conv2D
        # module.net[3] --> GroupNorm
        # module.net[4] --> Conv2D
        
        prune.ln_structured(module.net[1], 'weight', amount=pruning_percentage, dim=0, n=norm_n)
        # prune.ln_structured(module.net[3], 'weight', amount=pruning_percentage, dim=1, n=norm_n)
        prune.ln_structured(module.net[4], 'weight', amount=pruning_percentage, dim=1, n=norm_n)


def prune_remove(model):
    unet = model.modelDecoder
    for name, module in unet.named_modules():
        if type(module) is ConvNextBlock:
            print(name)
            prune.remove(module.net[1], 'weight')
            prune.remove(module.net[4], 'weight')


def evaluate_sparcity(model):
    total_params = 0
    total_zeros = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            total_zeros += torch.sum(module.weight == 0)

    return total_params, total_zeros.item()


def train(modelName:str, trainSet:TurbulenceDataset, testSets:Dict[str,TurbulenceDataset],
        p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
        p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True, gpuID:str="0"):

    # DATA AND MODEL SETUP
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    logger = Logger(modelName, addNumber=True)
    model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    model.printModelInfo()
    criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr, weight_decay=p_t.weightDecay)
    lrScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p_t.expLrGamma)
    logger.setup(model, optimizer)


    logging.info("--" * 30)
    logging.info(f"Pruning params: ({p_t.prune_freq}, {p_t.prune_times}, {p_t.prune_perc}, {p_t.prune_type})")
    logging.info("--" * 30)

    transTrain = Transforms(p_d)
    trainSet.transform = transTrain
    trainSet.printDatasetInfo()
    trainSampler = RandomSampler(trainSet)
    #trainSampler = SubsetRandomSampler(range(2))
    trainLoader = DataLoader(trainSet, sampler=trainSampler,
                    batch_size=p_d.batch, drop_last=True, num_workers=4)
    trainHistory = LossHistory("_train", "Training", logger.tfWriter, len(trainLoader),
                                    0, 1, printInterval=1, logInterval=1, simFields=p_d.simFields)
    trainer = Trainer(model, trainLoader, optimizer, lrScheduler, criterion, trainHistory, logger.tfWriter, p_d, p_t)

    testers = []
    testHistories = []
    for shortName, testSet in testSets.items():
        p_d_test = copy.deepcopy(p_d)
        p_d_test.augmentations = ["normalize", "resize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        if p_d.sequenceLength[0] != p_d_test.sequenceLength[0]:
            p_d_test.batch = 1

        transTest = Transforms(p_d_test)
        testSet.transform = transTest
        testSet.printDatasetInfo()
        testSampler = SequentialSampler(testSet)
        #testSampler = SubsetRandomSampler(range(2))
        testLoader = DataLoader(testSet, sampler=testSampler,
                        batch_size=p_d_test.batch, drop_last=False, num_workers=4)
        testHistory = LossHistory(shortName, testSet.name, logger.tfWriter, len(testLoader),
                                    25, 25, printInterval=0, logInterval=0, simFields=p_d.simFields)
        tester = Tester(model, testLoader, criterion, testHistory, p_t)
        testers += [tester]
        testHistories += [testHistory]

    #if loadEpoch > 0:
    #    logger.resumeTrainState(loadEpoch)

    # TRAINING
    print('Starting Training')
    logger.saveTrainState(0)

    for tester in testers:
        tester.testStep(0)

    n_pruned = 0

    for epoch in range(0, p_t.epochs):
        trainer.trainingStep(epoch)
        logger.saveTrainState(epoch)

        for tester in testers:
            tester.testStep(epoch+1)

        trainHistory.updateAccuracy([p_d,p_t,p_l,p_me,p_md,p_ml], testHistories, epoch==p_t.epochs-1)

        if n_pruned < p_t.prune_times and epoch % p_t.prune_freq == 0:
            logging.info(f"[{n_pruned}] Pruning model ({p_t.prune_type}, {p_t.prune_perc})")

            _prune(model, p_t.prune_type, p_t.prune_perc)
            
            nparams, nzeros = evaluate_sparcity(model)
            logging.info(f"[{n_pruned}] Model sparcity: {round(100 * nzeros / nparams, 2)}%")

            n_pruned += 1

    # Make prune permanent
    prune_remove(model)
    logger.saveTrainState(epoch)

    logger.close()

    print('Finished Training')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convnext_mult', type=int,   default=1,     help='convnext_mult paramater to scale net size')
    parser.add_argument('--prune_freq',    type=int,   default=1,     help='Prunning frequency')
    parser.add_argument('--prune_times',   type=int,   default=10,    help='Number of epochs to prune')
    parser.add_argument('--prune_perc',    type=float, default=0.1,   help='Pruning percentage')
    parser.add_argument('--prune_type',    type=str,   default="L2",  help='Pruning type')
    parser.add_argument('--epochs',        type=int,   default=100,   help='Number of epochs to train')
    parser.add_argument('--xsize',         type=int,   default=128,   help='Input X size, if different than (128,64) data is scaled')
    parser.add_argument('--ysize',         type=int,   default=64,    help='Input Y size, if different than (128,64) data is scaled')
    parser.add_argument('--noLSIM',        type=bool,  default=False, help='If true, the LSIM loss is deactivated for training and testing')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    useGPU = True
    gpuID = "0"

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    ### UNET
    modelName = "2D_Inc/128_unet-m2"
    p_d = DataParams(batch=32, augmentations=["normalize", "resize"], sequenceLength=[2,2], randSeqOffset=True,
                dataSize=[args.xsize, args.ysize], dimension=2, simFields=["pres"], simParams=["rey"], normalizeMode="incMixed")
    # p_d = DataParams(batch=32, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
                # dataSize=[128,64], dimension=2, simFields=["pres"], simParams=["rey"], normalizeMode="incMixed")
    
    p_t = TrainingParams(epochs=args.epochs, lr=0.0001, noLSIM=args.noLSIM)
    p_l = LossParams(recMSE=0.0, predMSE=1.0)
    p_me = None
    p_md = ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.0, convnext_mult=args.convnext_mult)
    p_ml = None
    pretrainPath = ""

    trainSet = TurbulenceDataset("Training", ["data"], filterTop=["128_inc"], filterSim=[(10,81)], filterFrame=[(800,1300)],
                    sequenceLength=[p_d.sequenceLength], randSeqOffset=p_d.randSeqOffset, simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim")

    testSets = {
        "lowRey":
            TurbulenceDataset("Test Low Reynolds 100-200", ["data"], filterTop=["128_inc"], filterSim=[[82,84,86,88,90]],
                    filterFrame=[(1000,1150)], sequenceLength=[[60,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
        "highRey" :
            TurbulenceDataset("Test High Reynolds 900-1000", ["data"], filterTop=["128_inc"], filterSim=[[0,2,4,6,8]],
                    filterFrame=[(1000,1150)], sequenceLength=[[60,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
        "varReyIn" :
            TurbulenceDataset("Test Varying Reynolds Number (200-900)", ["data"], filterTop=["128_reyVar"], filterSim=[[0]],
                    filterFrame=[(300,800)], sequenceLength=[[250,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
    }

    p_t.prune_freq = args.prune_freq
    p_t.prune_times = args.prune_times
    p_t.prune_perc = args.prune_perc
    p_t.prune_type = args.prune_type

    train(
        modelName, trainSet, testSets, 
        p_d, p_t, p_l, p_me, p_md, p_ml, 
        pretrainPath=pretrainPath, useGPU=useGPU, gpuID=gpuID
    ) 
    
    # model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    # model.printModelInfo()
    # _prune(model, "L1", 0.9)
