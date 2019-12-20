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


extern class Camera;

struct para3
{
	float e[3];
};

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
	
	
	Camera** cudaCam;
	para3 Lookat;
	para3 Lookfrom;
	para3 Vup;
	float FocusDist;
	float Aperture;
	float Fov;

	int frameCount;

	int break_flag;
	int end_flag;


	float targetClipUpperbound;
	int targetSPP;

	

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
	void initialization();
	void setLabel(int index,int x, int y, std::string name);
	void setLabel(int index,int x, int y, std::string name,int length,int width);
public slots:
	void Startear();
	void refresh();
	void Stop();
	void ShowPara();
	void setParameter();
	void discardParameter();
	
	
private:
	QLabel* labParameter[1];
	QLineEdit* lineParameter[1];
	QPushButton* StartButton;
	QPushButton* StopButton;
	QLabel* Lab;
	QLabel* labTargetSPP;
	QLabel* labClipUpperbound;
	QLabel* labLookatX;
	QLabel* labLookatY;
	QLabel* labLookatZ;
	QLabel* labLookfromX;
	QLabel* labLookfromY;
	QLabel* labLookfromZ;
	QLabel* labVupX;
	QLabel* labVupY;
	QLabel* labVupZ;
	QLabel* labFocusDist;
	QLabel* labAperture;
	QLabel* labFov;
	QPushButton* Updater;
	QPushButton* Discarder;
	QTextEdit* logText;
	QLineEdit* paraTargetSPP;
	QLineEdit* paraClipUpperbound;
	QLineEdit* paraLookatX;
	QLineEdit* paraLookatY;
	QLineEdit* paraLookatZ;
	QLineEdit* paraLookfromX;
	QLineEdit* paraLookfromY;
	QLineEdit* paraLookfromZ;
	QLineEdit* paraVupX;
	QLineEdit* paraVupY;
	QLineEdit* paraVupZ;
	QLineEdit* paraFocusDist;
	QLineEdit* paraAperture;
	QLineEdit* paraFov;
	Ui::RTRendererCUDAQTClass ui;
};

