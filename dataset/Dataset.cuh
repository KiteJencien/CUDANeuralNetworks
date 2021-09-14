//
// Created by DanielSun on 9/9/2021.
//

#ifndef NETWORKS_DATASET_CUH
#define NETWORKS_DATASET_CUH

#include "TrainData.cuh"
#include "../../Networks/NetConfig.cuh"
#include <string>

//#pragma comment(lib,"opencv_core452.lib")
//#pragma comment(lib,"opencv_highgui452.lib")
//#pragma comment(lib,"opencv_imgproc452.lib")


class Dataset {
public:
    struct DatasetInstance{
        vector<TrainData*> data = vector<TrainData*>(DATA_SIZE);
    };

    struct BatchInstance{
        vector<TrainData*> data = vector<TrainData*>(BATCH_SIZE);
    };

    static void readDatasetMNIST(DatasetInstance *instance, const string& type);

};


#endif //NETWORKS_DATASET_CUH
