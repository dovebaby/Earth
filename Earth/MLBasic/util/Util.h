﻿#pragma once
#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2410d")
#pragma comment(lib, "opencv_highgui2410d")
#pragma comment(lib, "opencv_features2d2410d")
#pragma comment(lib, "opencv_ml2410d")
#pragma comment(lib, "opencv_nonfree2410d")
#pragma comment(lib, "opencv_imgproc2410d")
#else
#pragma comment(lib, "opencv_core2410")
#pragma comment(lib, "opencv_highgui2410")
#pragma comment(lib, "opencv_features2d2410")
#pragma comment(lib, "opencv_ml2410")
#pragma comment(lib, "opencv_nonfree2410")
#pragma comment(lib, "opencv_imgproc2410")
#endif

#include <direct.h>  
#include <stdio.h> 
#include <io.h>
#include <iostream>
#include <string>
#include <fstream>
#include <string>

class ImageUtil {
public:
	static cv::Mat ImageReSize(const cv::Mat& image, int weight = 255, int height = 255, bool scale = false)
	{
		cv::Mat unified_image;
		cv::Size size;
		if (scale) {
			size = cv::Size(weight, height);
		} else {
			int s = cv::min(image.rows, image.cols);
			float scale = 1.0*weight / s;
			size = cv::Size(image.cols  * scale, image.rows * scale);
		}
		cv::resize(image, unified_image, size);
		cv::medianBlur(unified_image, unified_image, 3);
		return unified_image;
	}
	static cv::Mat load(std::string image_file_path) {
		cv::Mat image;
		if (image_file_path.find(".jpg")) 
			image = cv::imread(image_file_path);
		return image;
	}
};

class Util {
public:

	static void ExportToCSVFile(const string& strFileName, const vector<vector<string>>& data)
	{
		if (data.empty()) return;
		ofstream csvIn(strFileName);
		for ( int row_num = 0; row_num < data.size(); ++ row_num) {
			string line = "";
			for ( int col_num = 0; col_num < data[row_num].size(); ++ col_num ) {
				if (col_num > 0) line.append(",");
				line.append(data[row_num][col_num]);
			}
			line.append("\n");
			if(csvIn.is_open()) 
			{
				csvIn << line;
			}
			else
			{
				cout<<"Error in opennig"<<strFileName<<endl;
			}				
		}
		csvIn.close();
	}

	static vector<vector<string>> ImportDataFromCSV(const string& strFileName)
	{
		vector<vector<string>> data;

		try
		{
			ifstream csvOut(strFileName);
			char rowContent[256];
			string strRow;
			while(!csvOut.eof())
			{
				csvOut.getline(rowContent,200);
				vector<string> vecRow=splitCSVRow(rowContent);
				if (vecRow.empty()) continue;
				data.push_back(vecRow);
			}
			csvOut.close();
		}
		catch (...)
		{
			return data;
		}

		return data;
	}


	static vector<string> splitCSVRow(const string& row)
	{
		vector<string> result;
		string oneCell="";
		for(int i=0;i<row.length();i++)
		{
			if(row[i]==',')
			{
				result.push_back(oneCell);
				oneCell="";					
			}
			else
			{
				oneCell+=row[i];
			}
		}
		result.push_back(oneCell);
		return result;
	}


	static std::string getRealPath() {
		#define MAX_PATH 100
		char buffer[MAX_PATH];   
		getcwd(buffer, MAX_PATH);
		std::string realPath(buffer);
		return realPath;
	}

	static void getFiles(string path, vector<string> &files, bool recursion = false){
		long hFile = 0;
		struct _finddata_t fileInfo;
		std::string pathName;

		if ((hFile = _findfirst(pathName.assign(path).append("\\*").c_str(), &fileInfo)) == -1) {
			return ;
		}
		do {
			string subPath = pathName.assign(path);
			if (subPath.back()!='\\' && subPath.back()!='/') {
				subPath.append("\\");
			}
			subPath = subPath + fileInfo.name;
			if (fileInfo.attrib&_A_SUBDIR) {
				string name(fileInfo.name);
				if (recursion && (name!=".") && (name!="..")) getFiles(subPath, files, recursion);
				continue;
			}
			files.push_back(subPath);
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
		return;
	}

	static void getDataAndLabels(const string &data_dir, const string &path_label_file) {
		long hFile = 0;
		struct _finddata_t fileInfo;
		std::string pathName;

		if ((hFile = _findfirst(pathName.assign(data_dir).append("\\*").c_str(), &fileInfo)) == -1) {
			return ;
		}

		vector<vector<string>> pathandLabel;
		int flag=1;

		do {
			string subPath = pathName.assign(data_dir);
			vector<string> files;
			if (subPath.back()!='\\' && subPath.back()!='/') {
				subPath.append("\\");
			}
			subPath = subPath + fileInfo.name;
			if (fileInfo.attrib&_A_SUBDIR) {
				string name(fileInfo.name);
				if ((name!=".") && (name!="..")) 
				{
					getFiles(subPath, files,false);
					if(files.size()>0)
					{
						for(int i=0;i<files.size();i++)
						{
							vector<string> temp;
							temp.push_back(files[i]);
							temp.push_back(to_string((long long) flag));
							pathandLabel.push_back(temp);
						}
						flag++;
					}
				}
			}
			
		} while (_findnext(hFile, &fileInfo) == 0);
		_findclose(hFile);
		
		ExportToCSVFile(path_label_file,pathandLabel);
	}

	static void writeMat(const Mat &mat, const string &fileName)
	{
		int n = mat.rows;
		int m = mat.cols;

		ofstream fout(fileName);
		

		Mat_<uchar> mat_ = mat;
		
		fout << n  << " " << m << endl;
		
		cout << "SB" << endl;
		cout << mat_.at<uchar>(0, 0) << endl;
		//getchar();
		uchar* matData = mat.data;

		int x = matData[0];
		cout << x << endl;
		
		for(int i = 0; i < n; ++ i)
		{
			for(int j = 0; j < m; ++ j)
			{
				if(j > 0) fout << " ";				
				fout << fixed << setprecision(5) <<  matData[i * m + j];

				int x = matData[i * m + j];
				cout << "X" << x << endl;
				
m 
			}
			fout << endl;
		}
		fout.close();


	}
};

#endif