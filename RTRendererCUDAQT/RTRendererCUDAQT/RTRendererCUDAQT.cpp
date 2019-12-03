#include "RTRendererCUDAQT.h"
#include <QImage>
#include <QPainter>
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include "consts.h"



RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(90, 90, MAX_X + 100, MAX_Y + 100);
	pb = new QPushButton("Modify", this);
	pb->setGeometry(10, 10, 100, 20);
	lab = new QLabel(this);
	lab->setGeometry(10, 40 , MAX_X, MAX_Y);
	connect(pb, SIGNAL(clicked()), this, SLOT(SlotTest()));
}

void RTRendererCUDAQT::SlotTest()
{
	kernel();
}

