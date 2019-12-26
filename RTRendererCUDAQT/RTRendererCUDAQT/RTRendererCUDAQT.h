#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_RTRendererCUDAQT.h"
#include <qwidget.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qimage.h>
#include <qthread.h>
#include <qtextedit.h>
#include <qcombobox.h>
#include <qvariant.h>
#include <QPainter>
#include "consts.h"
#include <string>
#include <sstream>
#include <algorithm>
#include "Vec3.h"
#include <qmouseeventtransition.h>


extern class Camera;


//统一flag为0正常，为1退出

class LoopThread : public QThread
{
	Q_OBJECT
public:
	LoopThread(QObject* parent = 0)
		: QThread(parent)
	{
		;
		}
	void kernel();

	//对应每个物体是否显示
	int flag_show[OBJECT_NUM];

	std::string object_names[OBJECT_NUM];

	//各种参数

	//Camera
	Camera** cudaCam;
	Vec3* Lookat;
	Vec3* Lookfrom;
	Vec3* Vup;
	float FocusDist;
	float Aperture;
	float Fov;


	//Render
	float targetClipUpperbound;
	int targetSPP;


	//World
	float placeHolder;

	//当前帧数
	int frameCount;

	//控制信号：是否退出当前渲染
	int break_flag;
	//控制信号：是否退出程序
	int end_flag;

	int pause_flag;

	void PrintMessege();
	void showParameter();
	void checkBreak();
	
protected:
	void run();

signals:
	void done();
	void refresh_flag();
	void info_flag();
	void discard_flag();
	
};



class RTRendererCUDAQT : public QMainWindow
{
	Q_OBJECT

public:
	RTRendererCUDAQT(QWidget* parent = Q_NULLPTR);

	
	float clip_upperbound = 1;

	//渲染引擎线程
	LoopThread* looper;

	//Camera,World,Render的参数个数
	int paraNumCamere = 12;
	int paraNumWorld = 1;
	int paraNumRender = 2;

	//修改参数时的统一接口
	int changingNum;
	int selected;
	QLabel** changingLab;
	QLineEdit** changingLine;

	//集成的参数窗口创建函数
	void setLabelRender(int index,int x, int y, std::string name);
	void setLabelWorld(int index, int x, int y, std::string name);
	void setLabelCamera(int index, int x, int y, std::string name);

	void checkParameterLegal();

	void hideAll();
	void initialization();
	
public slots:
	//开始
	void Startear();

	//刷新显示
	void refresh();

	//退出
	void Stop();

	void Pause();

	void Reset();

	//显示参数
	void ShowPara();

	//设置参数
	void setParameter();

	//还原参数
	void discardParameter();

	//更改当前要修改的参数
	void changeParaCamera();
	void changeParaWorld();
	void changeParaRender();
	void changeObjectWorld();
	void changePara();
	void choosePara(int index);
	void chooseObject(int index);

	//物体的消失与加载
	void disappear();
	void appear();
	void showObject();
	
private:
	QLabel* Lab;
	QComboBox* Parameter;
	QComboBox* World;
	QLabel* labParaCamera[CAMERA_NUM];
	QLineEdit* lineParaCamera[CAMERA_NUM];
	QLabel* labParaWorld[WORLD_NUM];
	QLineEdit* lineParaWorld[WORLD_NUM];
	QLabel* labParaRender[RENDER_NUM];
	QLineEdit* lineParaRender[RENDER_NUM];
	QLineEdit* lineObjectName;
	QLabel* labObjectName;
	QLineEdit* lineObjectStatus;
	QLabel* labObjectStatus;
	QPushButton* AppearButton;
	QPushButton* DisappearButton;
	QPushButton* ResetButton;
	QPushButton* StartButton;
	QPushButton* ExitButton;
	QPushButton* PauseButton;
	QPushButton* Updater;
	QPushButton* Discarder;
	QTextEdit* logText;

	Ui::RTRendererCUDAQTClass ui;
};

