#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_RTRendererCUDAQT.h"
#include <qwidget.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qimage.h>
#include "consts.h"



extern void kernel();

class RTRendererCUDAQT : public QMainWindow
{
	Q_OBJECT

public:
	RTRendererCUDAQT(QWidget* parent = Q_NULLPTR);
	void kernel();
	double clip_upperbound = 1;
public slots:
	void SlotTest();
private:
	QPushButton* pb;
	QLabel* lab;
	Ui::RTRendererCUDAQTClass ui;
};
