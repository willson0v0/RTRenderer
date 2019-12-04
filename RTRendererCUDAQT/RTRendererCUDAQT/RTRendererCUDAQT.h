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

//extern void kernel();

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
	int break_flag = 0;
	int pre_break = 0;//默认是0，不依靠它提供break
	void PrintMessege();
protected:
	void run();

signals:
	void done();
	void refresh_flag();
	void info_flag();
};


class RTRendererCUDAQT : public QMainWindow
{
	Q_OBJECT

public:
	RTRendererCUDAQT(QWidget* parent = Q_NULLPTR);
	double clip_upperbound = 1;
	LoopThread* looper;
	
public slots:
	void Startear();
	void refresh();
	void Stop();
	void ShowPara();
	
	
private:
	QPushButton* StartButton;
	QPushButton* StopButton;
	QLabel* Lab;
	QPushButton* Updater;
	QTextEdit* ParameterText;
	QLineEdit* Para_Stop;
	Ui::RTRendererCUDAQTClass ui;
};

