#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_RTRendererCUDAQT.h"

extern "C" void launchKernal();

class RTRendererCUDAQT : public QMainWindow
{
	Q_OBJECT

public:
	RTRendererCUDAQT(QWidget *parent = Q_NULLPTR);

private:
	Ui::RTRendererCUDAQTClass ui;
};
