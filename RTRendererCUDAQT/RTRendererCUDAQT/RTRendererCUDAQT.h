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

	int frameCount;

	int break_flag;
	int end_flag;

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
	LoopThread* looper;
	int paraNumCamere = 12;
	int paraNumWorld = 1;
	int paraNumRender = 2;
	int changingNum;
	QLabel** changingLab;
	QLineEdit** changingLine;

	void setLabelRender(int index,int x, int y, std::string name);
	void setLabelWorld(int index, int x, int y, std::string name);
	void setLabelCamera(int index, int x, int y, std::string name);
	void hideAll();
	void initialization();
public slots:
	void Startear();
	void refresh();
	void Stop();
	void ShowPara();
	void setParameter();
	void discardParameter();
	void changeParaCamera();
	void changeParaWorld();
	void changeParaRender();
	void changePara();
	void choosePara(int index);
	
private:
	QLabel* Lab;
	QComboBox* Parameter;
	QLabel* labParaCamera[12];
	QLineEdit* lineParaCamera[12];
	QLabel* labParaWorld[1];
	QLineEdit* lineParaWorld[1];
	QLabel* labParaRender[2];
	QLineEdit* lineParaRender[2];
	QPushButton* StartButton;
	QPushButton* StopButton;
	QPushButton* Updater;
	QPushButton* Discarder;
	QTextEdit* logText;

	Ui::RTRendererCUDAQTClass ui;
};

