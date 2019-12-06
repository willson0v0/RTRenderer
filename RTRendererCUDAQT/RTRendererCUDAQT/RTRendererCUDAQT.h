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

//合并的话会发生什么？

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
	
	int frameCount;

	int break_flag = 0;
	bool reset_flag;


	float targetClipUpperbound = 1.0;
	int targetSPP = INT_MAX;//默认是最大值，不依靠它提供break

	float lookatX = 0;
	float lookatY = 0;
	float lookatZ = 0;

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
	QPushButton* Updater;
	QPushButton* Discarder;
	QTextEdit* logText;
	QLineEdit* paraTargetSPP;
	QLineEdit* paraClipUpperbound;
	Ui::RTRendererCUDAQTClass ui;
};

