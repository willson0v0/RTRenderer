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
extern class Vec3;

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
	float lookatX;
	float lookatY;
	float lookatZ;

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
	
public slots:
	void Startear();
	void refresh();
	void Stop();
	void ShowPara();
	void setParameter();
	void discardParameter();
	
	
private:
	QPushButton* StartButton;
	QPushButton* StopButton;
	QLabel* Lab;
	QLabel* labClipUpperbound;
	QLabel* labTargetSPP;
	QLabel* labLookatX;
	QLabel* labLookatY;
	QLabel* labLookatZ;
	QPushButton* Updater;
	QPushButton* Discarder;
	QTextEdit* logText;
	QLineEdit* paraTargetSPP;
	QLineEdit* paraClipUpperbound;
	QLineEdit* paraLookatX;
	QLineEdit* paraLookatY;
	QLineEdit* paraLookatZ;
	Ui::RTRendererCUDAQTClass ui;
};

