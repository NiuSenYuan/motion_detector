#include "motion_detector.h"
#include <iostream>
#include <thread>
#include <chrono>

MotionDetector::MotionDetector()
{
    rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
}

void MotionDetector::processVideo(const std::string &filename)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频文件：" << filename << std::endl;
        return;
    }

    cv::Mat frame, gray1;
    cap >> frame;
    if (frame.empty())
    {
        std::cerr << "无法读取视频第一帧！" << std::endl;
        return;
    }
    cv::cvtColor(frame, gray1, cv::COLOR_BGR2GRAY);

    while (true)
    {
        cv::Mat frame2, gray2;
        cap >> frame2;
        if (frame2.empty())
            break;

        /*
        实现彩色图象转灰度图像：
        彩色图像通常使用 BGR 格式，有 3 个通道（蓝、绿、红），而灰度图像只有 1 个通道（亮度）。
        将彩色图像转换为灰度图像，处理的数据量减少到三分之一，极大提高了：
        图像差分处理速度（帧差法），阈值操作、形态学操作、轮廓提取等步骤的效率
        */
        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

        /*
        帧差法处理流程：
        目的是检测出图像中发生运动（变化）的区域。
        定义两个图像矩阵：diff用来保存两帧图像的差值图，thresh用来保存经过二值化处理后的图像。
        1.absdiff用来计算两种灰度图gray1和gray2的逐像素的绝对差。
        本质是检测相邻帧之间的像素亮度变化程度，如果一个像素在两帧中变化不大，说明画面没动；如果变化很大，
        说明画面有运动。
        核心思想：帧差法 → 移动的物体在两帧之间会造成亮度差异。
        2.threshold:作用：将差值图 diff 转换为二值图 thresh
        参数说明：
        10：阈值，只有亮度差大于10的像素被认为“有变化”
        255：输出的白色像素值
        cv::THRESH_BINARY：应用标准的二值化操作（大于阈值的设为255，小于设为0）
        3.dilate(thresh, thresh, rectKernel, cv::Point(-1, -1), 2);
        作用：对二值图进行膨胀操作（dilation）
        参数说明：
        rectKernel：膨胀使用的核（形状一般为矩形）
        cv::Point(-1, -1)：锚点默认居中
        2：膨胀迭代次数
        膨胀的目的是：
        去掉白色区域中的小黑洞（连通区域更清晰）
        扩大白色目标区域，便于后续轮廓提取
        */
        cv::Mat diff, thresh;
        cv::absdiff(gray1, gray2, diff);
        cv::threshold(diff, thresh, 10, 255, cv::THRESH_BINARY);
        cv::dilate(thresh, thresh, rectKernel, cv::Point(-1, -1), 2);

        /*
        轮廓提取：目的是从处理后的二值图像中找出前景物体（例如运动目标）的边界轮廓。
        定义一个二维向量 contours，每个元素是一个轮廓（即点的集合）。
        一个轮廓（std::vector<cv::Point>）表示图像中一块连通区域的边界点列表。
        定义一个包含层次结构信息的向量 hierarchy，每个元素是一个长度为 4 的向量（Vec4i），表示轮廓的父子关系。
        常用于有嵌套关系的轮廓结构（例如“洞中有洞”的场景），虽然在 RETR_EXTERNAL 模式下不太用到。
        cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        是OpenCV中用于提取图像轮廓的函数。参数说明如下：
        | 参数                    | 含义                                                      |
        | ------------------------| ---------------------------------------------------------|
        | thresh                  | 输入图像，必须是 二值图（黑白图，前面用 threshold() 得到的） |
        | contours                | 输出参数，保存检测到的所有轮廓（每个轮廓是点的集合）          |
        | hierarchy               | 输出参数，保存轮廓之间的结构信息（父轮廓、子轮廓等）          |
        | cv::RETR_EXTERNAL       | 只提取 最外层轮廓（忽略内嵌轮廓），适合用于运动目标检测       |
        | cv::CHAIN_APPROX_SIMPLE | 使用边界压缩算法，只保存轮廓上的关键点，减少内存占用          |
        */
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        /*
        从所有检测到的轮廓中找出最大的一个（面积最大），然后用绿色矩形将其框选出来显示在图像上。
        使用 std::max_element() 找出 contours 中面积最大的那个轮廓。
        cv::contourArea(c)：计算一个轮廓 c 的面积。
        比较逻辑是：如果 c1 的面积小于 c2，就认为 c2 更“大”，从而最终找到最大面积的轮廓。
💡      注意：*std::max_element(...) 得到的是最大轮廓本身（类型是 std::vector<cv::Point>），命名为 largestContour。
        */
        if (!contours.empty())
        {
            auto largestContour = *std::max_element(contours.begin(), contours.end(),
                                                    [](const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2)
                                                    {
                                                        return cv::contourArea(c1) < cv::contourArea(c2);
                                                    });

            /*
            使用 OpenCV 提供的 cv::boundingRect() 函数对最大轮廓进行外接矩形拟合；
            得到的 boundRect 是一个 cv::Rect 对象，表示矩形的左上角坐标和宽高；
            它相当于用一个最小的“盒子”将这个轮廓框住。
            */
            cv::Rect boundRect = cv::boundingRect(largestContour);
            // 在原图 frame2 上绘制这个矩形框；
            cv::rectangle(frame2, boundRect, cv::Scalar(0, 255, 0), 2);
        }
        // 显示图像：将两个图像横向拼接后显示出来
        cv::Mat threshColor;
        cv::cvtColor(thresh, threshColor, cv::COLOR_GRAY2BGR);
        cv::Mat combined;
        cv::hconcat(frame2, threshColor, combined);
        cv::imshow("Difference", combined);

        gray1 = gray2.clone();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}
