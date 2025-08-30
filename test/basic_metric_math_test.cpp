#include "../src/ssimu2/main.hpp"
#include "../src/butter/main.hpp"

template <InputMemType T>
class Image{
private:
    uint8_t* srcp[3] = {NULL, NULL, NULL};
public:
    const uint8_t* csrcp[3];
    uint64_t width;
    uint64_t height;
    uint64_t stride;
    Image(uint64_t width, uint64_t height, uint64_t stride) : width(width), height(height), stride(stride){
        srcp[0] = (uint8_t*)malloc(stride*height*3); //stride takes into account type of T
        srcp[1] = srcp[0] + stride*height;
        srcp[2] = srcp[1] + stride*height;
        csrcp[0] = srcp[0];
        csrcp[1] = srcp[1];
        csrcp[2] = srcp[2];
    }
    void fillZero(){
        for (uint64_t i = 0; i < stride*height*3; i++){
            srcp[0][i] = 0;
        }
    }
    void fillZeroStrideOne(){
        fillZero();
        for (uint64_t j = 0; j < 3*height; j++){
            for (uint64_t i = width*sizeof(T); i < stride; i++){
                srcp[0][j*stride+i] = (uint8_t)255;
            }
        }
    }
    ~Image(){
        if (srcp[0]) free(srcp[0]);
        srcp[0] = NULL;
    }
};

int main(){
    Image<FLOAT> im1(5076, 9400, 30000);
    Image<FLOAT> im2(5076, 9400, 30000);
    Image<FLOAT> imb1(1076, 2400, 10000);
    Image<FLOAT> imb2(1076, 2400, 10000);
    ssimu2::SSIMU2ComputingImplementation ssimu2process;
    ssimu2process.init(5076, 9400);
    butter::ButterComputingImplementation butterprocess;
    butterprocess.init(1076, 2400, 203);

    std::cout << "Testing basic empty images with different stride data" << std::endl;
    im1.fillZero();
    im2.fillZeroStrideOne();
    imb1.fillZero();
    imb2.fillZeroStrideOne();

    std::cout << "SSIMU2..." << std::endl;
    double res = ssimu2process.run<FLOAT>(im1.csrcp, im2.csrcp, 30000);
    if (res != 100.){
        std::cout << "|Error| : failed previous test with " << res << " instead of 100." << std::endl;
        return 0;
    }
    std::cout << "Butter..." << std::endl;
    std::tuple<float, float, float> resb = butterprocess.run<FLOAT>(NULL, 0, imb1.csrcp, imb2.csrcp, 10000);
    if (std::get<0>(resb) != 0 || std::get<1>(resb) != 0. || std::get<2>(resb) != 0.){
        std::cout << "|Error| : failed previous test with " << std::get<0>(resb) << ", " << std::get<1>(resb) << " and " << std::get<2>(resb) << " instead of 0., 0. and 0." << std::endl;
        return 0;
    }

    ssimu2process.destroy();
    butterprocess.destroy();
    std::cout << "Passed All Tests" << std::endl;
    return;
}