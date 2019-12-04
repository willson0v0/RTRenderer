#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QApplication>
#include "ui_RTRendererCUDAQT.h"
#include <qwidget.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qimage.h>
#include <qthread.h>
#include <QImage>
#include <QPainter>


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
protected:
	void run();

signals:
	void done();
	void refresh_flag();
};


class RTRendererCUDAQT : public QMainWindow
{
	Q_OBJECT

public:
	RTRendererCUDAQT(QWidget* parent = Q_NULLPTR);
	double clip_upperbound = 1;
	LoopThread* looper;
	
public slots:
	void SlotTest();
	void refresh();
	void Stop();
	
private:
	QPushButton* pb;
	QPushButton* stop_pb;
	QLabel* lab;
	Ui::RTRendererCUDAQTClass ui;
};

